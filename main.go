// main.go — Program entry point and main input loop.
//
// Build:  go build .
// Run:    ./mandelbrot
//
// All fractal math lives in render.go.
// All display logic lives in terminal.go.
// Palette/color code lives in palette.go.
// Minibrot search lives in finder.go.
// File export lives in export.go.
// Shared globals live in globals.go.
package main

import (
	"bufio"
	"fmt"
	"math/big"
	"os"
	"runtime"
)

// shiftKeys maps shift+digit terminal bytes to bookmark slot numbers 1–9.
var shiftKeys = map[byte]int{
	'!': 1, '@': 2, '#': 3, '$': 4, '%': 5,
	'^': 6, '&': 7, '*': 8, '(': 9,
}

func main() {
	runtime.GOMAXPROCS(runtime.NumCPU())
	initLUTs()
	initTerminal()
	defer restoreTerminal()

	// Initial view: classic overview of the full Mandelbrot set.
	cx = new(big.Float).SetPrec(100).SetFloat64(-0.75)
	cy = new(big.Float).SetPrec(100).SetFloat64(0)
	zoom = new(big.Float).SetPrec(100).SetFloat64(1)

	for {
		// Auto-scale maxIter with zoom depth when enabled.
		if adaptIter {
			exp := zoom.MantExp(nil)
			if s := suggestMaxIter(exp); s > maxIter {
				maxIter = s
			}
		}

		drawTerminal()
		c := getChar()

		// ── Quit ────────────────────────────────────────────────────────────
		if c == 'q' || c == 3 { // 3 = ctrl-c
			fmt.Print("\r\n\033[0m\033[KQuit? (y/n): ")
			if ans := getChar(); ans == 'y' || ans == 'Y' {
				restoreTerminal()
				fmt.Print("\033[0m\033[H\033[J")
				os.Exit(0)
			}
			continue
		}

		// ── Bookmark load (keys 1–9) ─────────────────────────────────────────
		if c >= '1' && c <= '9' {
			loadBookmark(int(c - '0'))
			continue
		}
		// ── Bookmark save (shift+1 through shift+9) ──────────────────────────
		if slot, ok := shiftKeys[c]; ok {
			saveBookmark(slot)
			continue
		}

		// Movement step size: 20% of the visible width at the current zoom.
		moveAmount := new(big.Float).Quo(big.NewFloat(0.2), zoom)

		switch c {

		// ── Navigation ────────────────────────────────────────────────────────
		case 'w':
			cy.Sub(cy, moveAmount)
		case 's':
			cy.Add(cy, moveAmount)
		case 'a':
			cx.Sub(cx, moveAmount)
		case 'd':
			cx.Add(cx, moveAmount)

		// ── Zoom ──────────────────────────────────────────────────────────────
		case 'z':
			zoom.Mul(zoom, big.NewFloat(1.5))
		case 'x':
			zoom.Quo(zoom, big.NewFloat(1.5))
		case 'Z':
			zoom.Mul(zoom, big.NewFloat(10))
		case 'X':
			zoom.Quo(zoom, big.NewFloat(10))

		// ── Reset — requires confirmation so deep zooms aren't lost by accident ──
		case 'r':
			restoreTerminal()
			exp := zoom.MantExp(nil)
			depth10 := int(float64(exp) * 0.30103)
			fmt.Printf("\033[2K\rReset from 10^%d to start? (y/N): ", depth10)
			if ans := getChar(); ans == 'y' || ans == 'Y' {
				cx.SetFloat64(-0.75)
				cy.SetFloat64(0)
				zoom.SetFloat64(1)
				maxIter = 500
			}
			initTerminal()

		// ── Iteration count ───────────────────────────────────────────────────
		case 'i':
			maxIter *= 2
		case 'o':
			maxIter /= 2
			if maxIter < 50 {
				maxIter = 50
			}
		case 'I':
			maxIter *= 8
		case 'O':
			maxIter /= 8
			if maxIter < 50 {
				maxIter = 50
			}

		// ── Color density ─────────────────────────────────────────────────────
		case 'k':
			colorDensity *= 1.2
		case 'l':
			colorDensity /= 1.2
		case 'K':
			colorDensity *= 5
		case 'L':
			colorDensity /= 5

		// ── Palette ───────────────────────────────────────────────────────────
		case 'c':
			idx := 0
			for i, k := range paletteKeys {
				if k == currentPaletteName {
					idx = i
					break
				}
			}
			currentPaletteName = paletteKeys[(idx+1)%len(paletteKeys)]

		// ── Feature toggles ───────────────────────────────────────────────────
		case 'A':
			adaptIter = !adaptIter
		case 'e':
			histoEQ = !histoEQ

		// ── Julia mode ────────────────────────────────────────────────────────
		case 'J':
			juliaMode = !juliaMode
		case ',':
			juliaR -= 0.01
		case '.':
			juliaR += 0.01
		case ';':
			juliaI -= 0.01
		case '\'':
			juliaI += 0.01

		// ── Search / warp ─────────────────────────────────────────────────────
		case 'v':
			autoFindMinibrot()
		case 'g':
			iFeelLucky()

		// ── Help ──────────────────────────────────────────────────────────────
		case 'h':
			showHelp = !showHelp

		// ── Export ────────────────────────────────────────────────────────────
		case 'P':
			savePPM()
		case 'p':
			saveCurrentPNG()
		case 'M':
			exportAnimation(+1) // zoom-out
		case 'F':
			exportAnimation(-1) // zoom-in

		// ── Custom palette ────────────────────────────────────────────────────
		case 'n':
			restoreTerminal()
			fmt.Print("\033[2K\rHex colours (e.g. #ff0000,#00ff00,#0000ff): ")
			reader := bufio.NewReader(os.Stdin)
			input, _ := reader.ReadString('\n')
			parseHexPalette(input)
			initTerminal()
		}
	}
}

