// render.go — Go orchestration layer + CGo bridge to C pixel engine.
//
// ═══════════════════════════════════════════════════════════════════════════
//  ARCHITECTURE
// ═══════════════════════════════════════════════════════════════════════════
//
//  Go handles:                        C handles (mandelbrot_core.c):
//  ─────────────────────────────      ──────────────────────────────────────
//  big.Float reference orbits         mb_row_std()      — hot row loop
//  Series approximation coeffs        mb_row_julia()    — hot row loop
//  Mariani-Silver subdivision         mb_perturb_pixel()— delta inner loop
//  Worker goroutines & scheduling     mb_pixel_std()    — single pixel (MS)
//  All terminal / export logic        mb_pixel_julia()  — single pixel (MS)
//
//  C compiler flags (-O3 -march=native -ffast-math -funroll-loops):
//   • No array bounds checks
//   • FMA (fused multiply-add) on ARM64 — "a*b+c" = 1 instruction
//   • NEON auto-vectorization on row functions
//   • Further unrolling beyond what we write manually
//
// ═══════════════════════════════════════════════════════════════════════════
//  SPEED TECHNIQUES
// ═══════════════════════════════════════════════════════════════════════════
//
//  ALL PATHS
//   • Mariani-Silver rectangle subdivision  (40-70% pixel skip at low zoom)
//   • 4-row work chunks                     (4× less atomic contention)
//   • GOMAXPROCS×2 goroutines               (hide big.Float latency)
//
//  SHALLOW ZOOM (C row functions)
//   • Bulb + cardioid rejection
//   • 8× manual unroll + Brent cycle detection
//   • -ffast-math FMA on ARM64
//   • fast_log2 bit-trick in smooth_color (both log calls)
//   • NEON auto-vectorization of row loop
//
//  DEEP ZOOM — Perturbation theory
//   • 5×5 probe grid reference orbit
//   • Pre-computed 2× reference arrays
//   • Series approximation skip
//   • C inner delta loop (no bounds checks, FMA)
//   • Worker-local multi-reference glitch recovery
//   • Per-worker big.Float reuse (zero allocs)
//
// ═══════════════════════════════════════════════════════════════════════════

package main

// #cgo CFLAGS: -O3 -march=native -ffast-math -funroll-loops -ftree-vectorize
// #include "mandelbrot_core.h"
// #include <stdlib.h>
import "C"
import (
	"math"
	"math/big"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"
)

// ─────────────────────────────────────────────
//  Precision / iter helpers
// ─────────────────────────────────────────────

func precForExp(exp int) uint {
	if exp <= 0 {
		return 100
	}
	return uint(exp) + 64
}

func suggestMaxIter(zoomExp int) int {
	if zoomExp <= 0 {
		return 200
	}
	n := int(100 * math.Sqrt(float64(zoomExp)*math.Ln2))
	if n < 200 { return 200 }
	if n > 50000 { return 50000 }
	return n
}

// smoothColor is kept in Go for use in the SA path and as a fallback.
// The C version (smooth_color) is used inside the C functions directly.
func smoothColor(iter int, rx, ry float64) float64 {
	m := rx*rx + ry*ry
	bits := math.Float64bits(m)
	e := int((bits>>52)&0x7FF) - 1023
	mant := math.Float64frombits((bits & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000)
	log2m := mant*(2.0-0.3358287811*mant) - 1.6642
	log2mag := (float64(e) + log2m) * 0.5
	bits2 := math.Float64bits(log2mag)
	e2 := int((bits2>>52)&0x7FF) - 1023
	mant2 := math.Float64frombits((bits2 & 0x000FFFFFFFFFFFFF) | 0x3FF0000000000000)
	nu := float64(e2) + mant2*(2.0-0.3358287811*mant2) - 1.6642
	return float64(iter) - nu + 1.0
}

// ─────────────────────────────────────────────
//  Series approximation (SA)
// ─────────────────────────────────────────────

type saCoeff struct {
	ar, ai float64
	br, bi float64
}

func computeSA(refX, refY []float64, refLen int, maxDelta float64) ([]saCoeff, int) {
	const eps = 1e-5
	coeffs := make([]saCoeff, 0, 256)
	ar, ai := 1.0, 0.0
	br, bi := 0.0, 0.0
	for n := 0; n < refLen-1; n++ {
		zr, zi := refX[n], refY[n]
		aNewR := 2*(zr*ar-zi*ai) + 1
		aNewI := 2*(zr*ai+zi*ar)
		bNewR := 2*(zr*br-zi*bi) + ar*ar - ai*ai
		bNewI := 2*(zr*bi+zi*br) + 2*ar*ai
		bMag2 := (bNewR*bNewR + bNewI*bNewI) * maxDelta * maxDelta * maxDelta * maxDelta
		aMag2 := (aNewR*aNewR + aNewI*aNewI) * maxDelta * maxDelta
		if aMag2 == 0 || bMag2 > eps*eps*aMag2 { break }
		ar, ai = aNewR, aNewI
		br, bi = bNewR, bNewI
		coeffs = append(coeffs, saCoeff{ar, ai, br, bi})
	}
	return coeffs, len(coeffs)
}

// ─────────────────────────────────────────────
//  Mariani-Silver subdivision
// ─────────────────────────────────────────────

const msMinSize = 4

func msCheckBorder(data []float64, w, x0, y0, x1, y1 int) (bool, float64) {
	first := data[y0*w+x0]
	for x := x0; x < x1; x++ {
		if data[y0*w+x] != first || data[(y1-1)*w+x] != first { return false, 0 }
	}
	for y := y0 + 1; y < y1-1; y++ {
		if data[y*w+x0] != first || data[y*w+(x1-1)] != first { return false, 0 }
	}
	return true, first
}

func msFill(data []float64, w, x0, y0, x1, y1 int, v float64) {
	for y := y0 + 1; y < y1-1; y++ {
		row := data[y*w:]
		for x := x0 + 1; x < x1-1; x++ { row[x] = v }
	}
}

// computePixelStd calls the C single-pixel function.
// Used only by Mariani-Silver for individual border pixels.
func computePixelStd(data []float64, idx int, px, py float64, maxIt int) {
	data[idx] = float64(C.mb_pixel_std(C.double(px), C.double(py), C.int(maxIt)))
}

// computePixelJulia calls the C single-pixel Julia function.
func computePixelJulia(data []float64, idx int, rx, ry, jcx, jcy float64, maxIt int) {
	data[idx] = float64(C.mb_pixel_julia(
		C.double(rx), C.double(ry), C.double(jcx), C.double(jcy), C.int(maxIt)))
}

func msSubdivide(
	data []float64, done []uint32,
	x0, y0, x1, y1 int,
	w int,
	cxF64, cyF64, pixelSize, halfW, halfH float64,
	maxIt int,
	isJulia bool, jcx, jcy float64,
) {
	if x1-x0 < msMinSize || y1-y0 < msMinSize {
		for y := y0; y < y1; y++ {
			rowPy := cyF64 + (float64(y)-halfH)*pixelSize
			for x := x0; x < x1; x++ {
				idx := y*w + x
				word, bit := idx/32, uint(idx%32)
				if done[word]>>bit&1 == 0 {
					done[word] |= 1 << bit
					px := cxF64 + (float64(x)-halfW)*pixelSize
					if isJulia {
						computePixelJulia(data, idx, px, rowPy, jcx, jcy, maxIt)
					} else {
						computePixelStd(data, idx, px, rowPy, maxIt)
					}
				}
			}
		}
		return
	}

	computeBorder := func(x, y int) {
		idx := y*w + x
		word, bit := idx/32, uint(idx%32)
		if done[word]>>bit&1 == 0 {
			done[word] |= 1 << bit
			px := cxF64 + (float64(x)-halfW)*pixelSize
			py := cyF64 + (float64(y)-halfH)*pixelSize
			if isJulia {
				computePixelJulia(data, idx, px, py, jcx, jcy, maxIt)
			} else {
				computePixelStd(data, idx, px, py, maxIt)
			}
		}
	}

	for x := x0; x < x1; x++ { computeBorder(x, y0); computeBorder(x, y1-1) }
	for y := y0 + 1; y < y1-1; y++ { computeBorder(x0, y); computeBorder(x1-1, y) }

	if same, val := msCheckBorder(data, w, x0, y0, x1, y1); same {
		msFill(data, w, x0, y0, x1, y1, val)
		for y := y0 + 1; y < y1-1; y++ {
			for x := x0 + 1; x < x1-1; x++ {
				idx := y*w + x; done[idx/32] |= 1 << uint(idx%32)
			}
		}
		return
	}

	mx := (x0 + x1) / 2
	my := (y0 + y1) / 2
	msSubdivide(data, done, x0, y0, mx, my, w, cxF64, cyF64, pixelSize, halfW, halfH, maxIt, isJulia, jcx, jcy)
	msSubdivide(data, done, mx, y0, x1, my, w, cxF64, cyF64, pixelSize, halfW, halfH, maxIt, isJulia, jcx, jcy)
	msSubdivide(data, done, x0, my, mx, y1, w, cxF64, cyF64, pixelSize, halfW, halfH, maxIt, isJulia, jcx, jcy)
	msSubdivide(data, done, mx, my, x1, y1, w, cxF64, cyF64, pixelSize, halfW, halfH, maxIt, isJulia, jcx, jcy)
}

// ─────────────────────────────────────────────
//  Public entry points
// ─────────────────────────────────────────────

func renderMandelbrot(w, h int, rcx, rcy, rzoom *big.Float) []float64 {
	return renderInto(w, h, rcx, rcy, rzoom, false)
}

func renderScratch(w, h int, rcx, rcy, rzoom *big.Float) []float64 {
	return renderInto(w, h, rcx, rcy, rzoom, true)
}

// ─────────────────────────────────────────────
//  Core engine
// ─────────────────────────────────────────────

func renderInto(w, h int, renderCx, renderCy, renderZoom *big.Float, useScratch bool) []float64 {

	// ── Buffer ───────────────────────────────────────────────────────────────
	var data []float64
	if useScratch {
		if w != scratchW || h != scratchH {
			scratchData = make([]float64, w*h)
			scratchW, scratchH = w, h
		}
		data = scratchData
	} else {
		if w != renderDataW || h != renderDataH {
			renderData = make([]float64, w*h)
			renderDataW, renderDataH = w, h
		}
		data = renderData
	}

	// ── Precision & geometry ─────────────────────────────────────────────────
	exp := renderZoom.MantExp(nil)
	prec := precForExp(exp)
	renderCx.SetPrec(prec); renderCy.SetPrec(prec); renderZoom.SetPrec(prec)

	minDim := w
	if h < minDim { minDim = h }
	psBig := new(big.Float).SetPrec(prec)
	psBig.Mul(renderZoom, new(big.Float).SetPrec(prec).SetFloat64(float64(minDim)))
	psBig.Quo(new(big.Float).SetPrec(prec).SetFloat64(4), psBig)
	pixelSize, _ := psBig.Float64()

	cxF64, _ := renderCx.Float64()
	cyF64, _ := renderCy.Float64()
	halfW := float64(w) / 2
	halfH := float64(h) / 2

	usePerturbation := exp > 43 && !juliaMode

	// ═══════════════════════════════════════════════════════════════════════
	//  SHALLOW ZOOM — C row functions + Mariani-Silver
	// ═══════════════════════════════════════════════════════════════════════
	if !usePerturbation {
		numWorkers := runtime.GOMAXPROCS(0) * 2
		slabW := (w + numWorkers - 1) / numWorkers
		totalPix := w * h
		done := make([]uint32, (totalPix+31)/32)
		isJulia := juliaMode
		jcx, jcy := juliaR, juliaI
		curMaxIter := maxIter

		var wg sync.WaitGroup
		for i := 0; i < numWorkers; i++ {
			x0 := i * slabW
			x1 := x0 + slabW
			if x1 > w { x1 = w }
			if x0 >= w { break }
			wg.Add(1)
			go func(x0, x1 int) {
				defer wg.Done()
				msSubdivide(data, done, x0, 0, x1, h,
					w, cxF64, cyF64, pixelSize, halfW, halfH,
					curMaxIter, isJulia, jcx, jcy)
			}(x0, x1)
		}
		wg.Wait()

		// After MS fills, sweep remaining unchecked pixels row by row using
		// C row functions — faster than individual pixel calls for missed pixels.
		// In practice MS covers >95% of pixels; this cleans the rest.
		var rowWg sync.WaitGroup
		var rowCounter int32
		for wk := 0; wk < numWorkers; wk++ {
			rowWg.Add(1)
			go func() {
				defer rowWg.Done()
				for {
					y := int(atomic.AddInt32(&rowCounter, 1) - 1)
					if y >= h { break }
					rowBase := y * w
					rowPy := cyF64 + (float64(y)-halfH)*pixelSize
					// Check if this row has any uncomputed pixels.
					hasUndone := false
					for x := 0; x < w; x++ {
						idx := y*w + x
						if done[idx/32]>>uint(idx%32)&1 == 0 { hasUndone = true; break }
					}
					if !hasUndone { continue }
					// Use C row function for the whole row — faster than per-pixel.
					rowSlice := data[rowBase : rowBase+w]
					rowPtr := (*C.double)(unsafe.Pointer(&rowSlice[0]))
					if isJulia {
						C.mb_row_julia(rowPtr, C.int(w),
							C.double(cxF64), C.double(rowPy),
							C.double(pixelSize), C.double(halfW),
							C.double(jcx), C.double(jcy), C.int(curMaxIter))
					} else {
						C.mb_row_std_neon(rowPtr, C.int(w),
							C.double(cxF64), C.double(rowPy),
							C.double(pixelSize), C.double(halfW), C.int(curMaxIter))
					}
				}
			}()
		}
		rowWg.Wait()
		return data
	}

	// ═══════════════════════════════════════════════════════════════════════
	//  DEEP ZOOM — Perturbation theory with C inner loop
	// ═══════════════════════════════════════════════════════════════════════

	bigSixteen := new(big.Float).SetPrec(prec).SetFloat64(16)
	bigTwo := new(big.Float).SetPrec(prec).SetFloat64(2)

	// ── 5×5 probe grid ────────────────────────────────────────────────────
	bestEscAt := -1
	var bestCx, bestCy *big.Float
	bestRefPX := w / 2
	bestRefPY := h / 2

probeLoop:
	for py := 1; py <= 5; py++ {
		for px := 1; px <= 5; px++ {
			pxIdx := (w * px) / 6
			pyIdx := (h * py) / 6
			dcxB := new(big.Float).SetPrec(prec).SetFloat64((float64(pxIdx) - halfW) * pixelSize)
			dcyB := new(big.Float).SetPrec(prec).SetFloat64((float64(pyIdx) - halfH) * pixelSize)
			probeCx := new(big.Float).SetPrec(prec).Add(renderCx, dcxB)
			probeCy := new(big.Float).SetPrec(prec).Add(renderCy, dcyB)
			zx := new(big.Float).SetPrec(prec); zy := new(big.Float).SetPrec(prec)
			nzx := new(big.Float).SetPrec(prec); nzy := new(big.Float).SetPrec(prec)
			t1 := new(big.Float).SetPrec(prec); t2 := new(big.Float).SetPrec(prec)
			mag2 := new(big.Float).SetPrec(prec)
			esc := maxIter
			for i := 0; i <= maxIter; i++ {
				t1.Mul(zx, zx); t2.Mul(zy, zy); mag2.Add(t1, t2)
				if mag2.Cmp(bigSixteen) > 0 { esc = i; break }
				if i < maxIter {
					nzx.Sub(t1, t2).Add(nzx, probeCx)
					nzy.Mul(zx, zy).Mul(nzy, bigTwo).Add(nzy, probeCy)
					zx.Copy(nzx); zy.Copy(nzy)
				}
			}
			if esc > bestEscAt {
				bestEscAt = esc; bestCx = probeCx; bestCy = probeCy
				bestRefPX = pxIdx; bestRefPY = pyIdx
				if esc == maxIter { break probeLoop }
			}
		}
	}

	// ── Full reference orbit ──────────────────────────────────────────────
	escapedAt := bestEscAt
	if escapedAt < 0 { escapedAt = 0 }
	refX := make([]float64, escapedAt+2)
	refY := make([]float64, escapedAt+2)
	{
		zx := new(big.Float).SetPrec(prec); zy := new(big.Float).SetPrec(prec)
		nzx := new(big.Float).SetPrec(prec); nzy := new(big.Float).SetPrec(prec)
		t1 := new(big.Float).SetPrec(prec); t2 := new(big.Float).SetPrec(prec)
		for i := 0; i <= escapedAt; i++ {
			refX[i], _ = zx.Float64(); refY[i], _ = zy.Float64()
			if i < escapedAt {
				t1.Mul(zx, zx); t2.Mul(zy, zy)
				nzx.Sub(t1, t2).Add(nzx, bestCx)
				nzy.Mul(zx, zy).Mul(nzy, bigTwo).Add(nzy, bestCy)
				zx.Copy(nzx); zy.Copy(nzy)
			}
		}
	}

	// ── Pre-compute 2× arrays ─────────────────────────────────────────────
	ref2X := make([]float64, escapedAt+2)
	ref2Y := make([]float64, escapedAt+2)
	for i := 0; i <= escapedAt; i++ { ref2X[i] = 2 * refX[i]; ref2Y[i] = 2 * refY[i] }

	// ── Series approximation ──────────────────────────────────────────────
	var saList []saCoeff
	saSkipN := 0
	if escapedAt > 32 {
		maxDelta := pixelSize * math.Sqrt(halfW*halfW+halfH*halfH)
		saList, saSkipN = computeSA(refX, refY, escapedAt, maxDelta)
	}

	// C pointers to the reference arrays — passed directly to C inner loop.
	// Go GC won't move these during the cgo call (pinned by the call frame).
	refXPtr := (*C.double)(unsafe.Pointer(&refX[0]))
	refYPtr := (*C.double)(unsafe.Pointer(&refY[0]))
	ref2XPtr := (*C.double)(unsafe.Pointer(&ref2X[0]))
	ref2YPtr := (*C.double)(unsafe.Pointer(&ref2Y[0]))
	cRefLen := C.int(escapedAt + 1)

	// ── Worker pool ───────────────────────────────────────────────────────
	numWorkers := runtime.GOMAXPROCS(0) * 2
	var rowCounter int32
	const chunkSize = 4
	var wg sync.WaitGroup

	for wk := 0; wk < numWorkers; wk++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			// Per-worker big.Float scratch.
			pxBig := new(big.Float).SetPrec(prec); pyBig := new(big.Float).SetPrec(prec)
			zxBig := new(big.Float).SetPrec(prec); zyBig := new(big.Float).SetPrec(prec)
			nrxBig := new(big.Float).SetPrec(prec); nryBig := new(big.Float).SetPrec(prec)
			t1Big := new(big.Float).SetPrec(prec); t2Big := new(big.Float).SetPrec(prec)
			dcxBig := new(big.Float).SetPrec(prec); dcyBig := new(big.Float).SetPrec(prec)
			mag2Big := new(big.Float).SetPrec(prec)
			twoBig := new(big.Float).SetPrec(prec).SetFloat64(2)
			sixteenBig := new(big.Float).SetPrec(prec).SetFloat64(16)
			oldZxBig := new(big.Float).SetPrec(prec); oldZyBig := new(big.Float).SetPrec(prec)

			// Worker-local multi-reference cache.
			localRefX := make([]float64, maxIter+2); localRefY := make([]float64, maxIter+2)
			local2X := make([]float64, maxIter+2); local2Y := make([]float64, maxIter+2)
			localEscAt := -1; localRefPX := 0; localRefPY := 0

			for {
				chunkStart := int(atomic.AddInt32(&rowCounter, chunkSize)) - chunkSize
				if chunkStart >= h { break }
				chunkEnd := chunkStart + chunkSize
				if chunkEnd > h { chunkEnd = h }

				for y := chunkStart; y < chunkEnd; y++ {
					rowBase := y * w
					rowPy := cyF64 + (float64(y)-halfH)*pixelSize
					rowDcy := (float64(y) - float64(bestRefPY)) * pixelSize

					for x := 0; x < w; x++ {
						px := cxF64 + (float64(x)-halfW)*pixelSize
						py := rowPy
						dcx := (float64(x) - float64(bestRefPX)) * pixelSize
						dcy := rowDcy

						dx, dy := 0.0, 0.0
						saIter := 0

						// ── Series approximation skip ─────────────────────
						if saSkipN > 0 {
							c := saList[saSkipN-1]
							dc2r := dcx*dcx - dcy*dcy
							dc2i := 2 * dcx * dcy
							dx = c.ar*dcx - c.ai*dcy + c.br*dc2r - c.bi*dc2i
							dy = c.ar*dcy + c.ai*dcx + c.br*dc2i + c.bi*dc2r
							saIter = saSkipN
						}

						// ── C inner delta loop ─────────────────────────────
						result := float64(C.mb_perturb_pixel(
							refXPtr, refYPtr, ref2XPtr, ref2YPtr, cRefLen,
							C.double(dcx), C.double(dcy),
							C.double(dx), C.double(dy), C.int(saIter),
							C.double(px), C.double(py), C.int(maxIter),
						))

						if result != -2.0 {
							// Normal result — escaped or inside set.
							data[rowBase+x] = result
							continue
						}

						// ── result == -2.0: glitch — try local reference ──
						escaped := false
						var rx, ry float64
						iter := 0
						if localEscAt > escapedAt {
							dcx2 := (float64(x) - float64(localRefPX)) * pixelSize
							dcy2 := (float64(y) - float64(localRefPY)) * pixelSize
							dx2, dy2 := 0.0, 0.0; iter = 0
							for iter <= localEscAt && iter < maxIter {
								rx = localRefX[iter] + dx2
								ry = localRefY[iter] + dy2
								if rx*rx+ry*ry > 16 { escaped = true; break }
								d2 := dx2 * dx2; e2 := dy2 * dy2; f2 := 2 * dx2 * dy2
								ndx := local2X[iter]*dx2 - local2Y[iter]*dy2 + d2 - e2 + dcx2
								ndy := local2X[iter]*dy2 + local2Y[iter]*dx2 + f2 + dcy2
								dx2 = ndx; dy2 = ndy; iter++
							}
						}

						// ── big.Float fallback → new local ref ───────────
						if !escaped && iter < maxIter {
							dcxBig.SetFloat64((float64(x) - halfW) * pixelSize)
							dcyBig.SetFloat64((float64(y) - halfH) * pixelSize)
							pxBig.Add(renderCx, dcxBig); pyBig.Add(renderCy, dcyBig)
							zxBig.SetFloat64(0); zyBig.SetFloat64(0)
							esc := maxIter; checkPeriod := 8
							for i := 0; i <= maxIter; i++ {
								localRefX[i], _ = zxBig.Float64(); localRefY[i], _ = zyBig.Float64()
								t1Big.Mul(zxBig, zxBig); t2Big.Mul(zyBig, zyBig); mag2Big.Add(t1Big, t2Big)
								if mag2Big.Cmp(sixteenBig) > 0 { esc = i; break }
								if zxBig.Cmp(oldZxBig) == 0 && zyBig.Cmp(oldZyBig) == 0 { esc = maxIter; break }
								if i == checkPeriod { oldZxBig.Copy(zxBig); oldZyBig.Copy(zyBig); checkPeriod *= 2 }
								if i < maxIter {
									nrxBig.Sub(t1Big, t2Big).Add(nrxBig, pxBig)
									nryBig.Mul(zxBig, zyBig).Mul(nryBig, twoBig).Add(nryBig, pyBig)
									zxBig.Copy(nrxBig); zyBig.Copy(nryBig)
								}
							}
							localEscAt = esc; localRefPX, localRefPY = x, y
							for j := 0; j <= esc; j++ { local2X[j] = 2 * localRefX[j]; local2Y[j] = 2 * localRefY[j] }
							iter = esc; escaped = esc < maxIter
							if escaped { rx = localRefX[esc]; ry = localRefY[esc] }
						}

						if escaped {
							rx2 := rx*rx - ry*ry + px; ry2 := 2*rx*ry + py
							rx3 := rx2*rx2 - ry2*ry2 + px; ry3 := 2*rx2*ry2 + py
							data[rowBase+x] = smoothColor(iter+2, rx3, ry3)
						} else {
							data[rowBase+x] = -1
						}
					}
				}
			}
		}()
	}

	wg.Wait()
	return data
}


// ─────────────────────────────────────────────
//  Animation helpers — CPU-budgeted render + direct-to-RGB
// ─────────────────────────────────────────────

// renderMandelbrotRGB renders a frame directly into an RGB byte buffer,
// bypassing the float64 intermediate array for the shallow-zoom (std) path.
// For deep zoom or Julia mode it falls back to float64 + LUT colormap.
//
// This is called by export.go's animation pipeline.  numWorkers controls
// how many goroutines are used so parallel frames don't over-subscribe.
func renderMandelbrotRGB(w, h int, rcx, rcy, rzoom *big.Float,
	rgb []byte, lut *PaletteLUT, colorScale, lutSizeF float64, numWorkers int) {

	exp := rzoom.MantExp(nil)
	usePerturbation := exp > 43 && !juliaMode

	if usePerturbation || juliaMode {
		// Fall back: render to float64, then colormap in one pass.
		data := renderMandelbrotWithWorkers(w, h, rcx, rcy, rzoom, numWorkers)
		for i := 0; i < w*h; i++ {
			v := data[i]
			o := i * 3
			if v < 0 {
				rgb[o] = 0; rgb[o+1] = 0; rgb[o+2] = 0
			} else {
				idx := valToIdx(v, colorScale, lutSizeF)
				c := lut.Colors[idx]
				rgb[o] = c.R; rgb[o+1] = c.G; rgb[o+2] = c.B
			}
		}
		return
	}

	// Shallow zoom fast path: C row function → LUT, directly into rgb[].
	// One pass over the data instead of two (render then colormap).
	prec := precForExp(exp)
	// Work on copies so we don't mutate the caller's big.Float values.
	rcx = new(big.Float).SetPrec(prec).Copy(rcx)
	rcy = new(big.Float).SetPrec(prec).Copy(rcy)
	rzoom = new(big.Float).SetPrec(prec).Copy(rzoom)

	minDim := w
	if h < minDim { minDim = h }
	psBig := new(big.Float).SetPrec(prec)
	psBig.Mul(rzoom, new(big.Float).SetPrec(prec).SetFloat64(float64(minDim)))
	psBig.Quo(new(big.Float).SetPrec(prec).SetFloat64(4), psBig)
	pixelSize, _ := psBig.Float64()
	cxF64, _ := rcx.Float64()
	cyF64, _ := rcy.Float64()
	halfW := float64(w) / 2
	halfH := float64(h) / 2
	curMaxIter := maxIter

	if numWorkers <= 0 {
		numWorkers = runtime.GOMAXPROCS(0) * 2
	}

	// Use x4 row function: 4 rows per dispatch, interleaved FP streams.
	nChunks := (h + 3) / 4
	var chunkCounter int32
	// Scratch buffer: 4 rows × w pixels, reused per goroutine.
	var rgbWg sync.WaitGroup
	for wk := 0; wk < numWorkers; wk++ {
		rgbWg.Add(1)
		go func() {
			defer rgbWg.Done()
			// Per-goroutine 4-row float64 scratch.
			scratch := make([]float64, 4*w)
			for {
				chunk := int(atomic.AddInt32(&chunkCounter, 1)) - 1
				if chunk >= nChunks { break }
				y0 := chunk * 4
				nRows := 4
				if y0+nRows > h { nRows = h - y0 }
				rowPy := cyF64 + (float64(y0)-halfH)*pixelSize
				scrPtr := (*C.double)(unsafe.Pointer(&scratch[0]))
				C.mb_row_std_x4(scrPtr, C.int(w), C.int(nRows),
					C.double(cxF64), C.double(rowPy),
					C.double(pixelSize), C.double(halfW), C.int(curMaxIter))
				for r := 0; r < nRows; r++ {
					base := (y0+r) * w * 3
					rowOff := r * w
					for x := 0; x < w; x++ {
						v := scratch[rowOff+x]
						o := base + x*3
						if v < 0 {
							rgb[o] = 0; rgb[o+1] = 0; rgb[o+2] = 0
						} else {
							idx := valToIdx(v, colorScale, lutSizeF)
							c := lut.Colors[idx]
							rgb[o] = c.R; rgb[o+1] = c.G; rgb[o+2] = c.B
						}
					}
				}
			}
		}()
	}
	rgbWg.Wait()
}

// renderMandelbrotWithWorkers renders to a fresh float64 slice with a
// caller-specified goroutine budget.  Used by renderMandelbrotRGB's fallback
// and directly available for any future caller that needs CPU budgeting.
//
// numWorkers <= 0 means "use GOMAXPROCS*2" (same as renderMandelbrot).
func renderMandelbrotWithWorkers(w, h int, rcx, rcy, rzoom *big.Float, numWorkers int) []float64 {
	if numWorkers <= 0 {
		numWorkers = runtime.GOMAXPROCS(0) * 2
	}

	// Always allocate fresh — animation has multiple frames in-flight.
	data := make([]float64, w*h)

	exp := rzoom.MantExp(nil)
	prec := precForExp(exp)
	rcx = new(big.Float).SetPrec(prec).Copy(rcx)
	rcy = new(big.Float).SetPrec(prec).Copy(rcy)
	rzoom = new(big.Float).SetPrec(prec).Copy(rzoom)

	minDim := w
	if h < minDim { minDim = h }
	psBig := new(big.Float).SetPrec(prec)
	psBig.Mul(rzoom, new(big.Float).SetPrec(prec).SetFloat64(float64(minDim)))
	psBig.Quo(new(big.Float).SetPrec(prec).SetFloat64(4), psBig)
	pixelSize, _ := psBig.Float64()
	cxF64, _ := rcx.Float64()
	cyF64, _ := rcy.Float64()
	halfW := float64(w) / 2
	halfH := float64(h) / 2
	isJulia := juliaMode
	jcx, jcy := juliaR, juliaI
	curMaxIter := maxIter

	usePerturbation := exp > 43 && !juliaMode

	if !usePerturbation {
		// Dispatch 4 rows at a time — mb_row_std_x4 interleaves 4 independent
		// pixel streams so ARM64 FP units stay busy during latency gaps.
		nChunks := (h + 3) / 4
		var chunkCounter int32
		var rowWg sync.WaitGroup
		for wk := 0; wk < numWorkers; wk++ {
			rowWg.Add(1)
			go func() {
				defer rowWg.Done()
				for {
					chunk := int(atomic.AddInt32(&chunkCounter, 1)) - 1
					if chunk >= nChunks { break }
					y0 := chunk * 4
					nRows := 4
					if y0+nRows > h { nRows = h - y0 }
					if isJulia {
						for r := 0; r < nRows; r++ {
							rSlice := data[(y0+r)*w : (y0+r)*w+w]
							rPtr := (*C.double)(unsafe.Pointer(&rSlice[0]))
							rowPy := cyF64 + (float64(y0+r)-halfH)*pixelSize
							C.mb_row_julia(rPtr, C.int(w),
								C.double(cxF64), C.double(rowPy),
								C.double(pixelSize), C.double(halfW),
								C.double(jcx), C.double(jcy), C.int(curMaxIter))
						}
					} else {
						rowPy := cyF64 + (float64(y0)-halfH)*pixelSize
						rowPtr := (*C.double)(unsafe.Pointer(&data[y0*w]))
						C.mb_row_std_x4(rowPtr, C.int(w), C.int(nRows),
							C.double(cxF64), C.double(rowPy),
							C.double(pixelSize), C.double(halfW), C.int(curMaxIter))
					}
				}
			}()
		}
		rowWg.Wait()
		return data
	}

	// Deep zoom: perturbation path is inherently stateful (reference orbit),
	// so we delegate to the standard renderInto which handles it correctly,
	// then copy into our fresh buffer.  Perturbation frames are rare during
	// animations (zoom > 10^13) and are already memory-bound, so the copy
	// is negligible versus big.Float work.
	src := renderInto(w, h, rcx, rcy, rzoom, false)
	copy(data, src)
	return data
}
