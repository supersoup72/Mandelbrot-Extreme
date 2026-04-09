// terminal.go — Terminal setup, teardown, size detection, and the main
// display loop that renders each frame to stdout as a single write.
package main

import (
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
)

// ─────────────────────────────────────────────
//  Terminal control
// ─────────────────────────────────────────────

// initTerminal puts the terminal into raw/cbreak mode so we can read
// individual keystrokes without waiting for Enter.
func initTerminal() {
	exec.Command("stty", "-F", "/dev/tty", "cbreak", "min", "1").Run()
	exec.Command("stty", "-F", "/dev/tty", "-echo").Run()
}

// restoreTerminal resets the terminal to its original "sane" state.
// Deferred in main() so it runs on any exit path.
func restoreTerminal() {
	exec.Command("stty", "-F", "/dev/tty", "sane").Run()
}

// getTermSize returns the terminal width and height in characters.
// Falls back to 80×24 on any error.
func getTermSize() (w, h int) {
	cmd := exec.Command("stty", "size")
	cmd.Stdin = os.Stdin
	out, err := cmd.Output()
	if err != nil {
		return 80, 24
	}
	parts := strings.Fields(string(out))
	if len(parts) < 2 {
		return 80, 24
	}
	rows, _ := strconv.Atoi(parts[0])
	cols, _ := strconv.Atoi(parts[1])
	if cols < 4 {
		cols = 80
	}
	if rows < 4 {
		rows = 24
	}
	return cols, rows
}

// getChar reads a single raw byte from stdin (blocking).
func getChar() byte {
	b := [1]byte{}
	os.Stdin.Read(b[:])
	return b[0]
}

// ─────────────────────────────────────────────
//  Frame renderer
// ─────────────────────────────────────────────

// drawTerminal renders one complete frame and writes it to stdout in a
// single os.Stdout.Write call — required by Termux to prevent tearing.
//
// Technique: Unicode LOWER HALF BLOCK (▄) packs two pixel rows into one
// terminal character row. Background colour = top pixel, foreground = bottom.
// ANSI codes are emitted only when the colour changes (run-length style),
// keeping stdout bytes minimal.
func drawTerminal() {
	w, rows := getTermSize()
	h := (rows - 2) * 2 // two pixel rows per char row; reserve 1 row for status
	if h < 2 {
		h = 2
	}

	data := renderMandelbrot(w, h, cx, cy, zoom)

	// Optional histogram equalization — remap before display only.
	displayData := data
	if histoEQ {
		if mapped := buildHistoMap(data); mapped != nil {
			displayData = mapped
		}
	}

	lut := luts[currentPaletteName]
	colorScale := colorDensity * 0.01 // hoisted out of per-pixel loop
	lutSizeF := float64(lutSize)       // hoisted: avoids int→float64 cast per pixel

	// ── Buffer management ───────────────────────────────────────────────────
	// termBuf capacity grows to fit and is retained across frames (zero GC).
	need := w*(h/2)*42 + 512
	termBuf = termBuf[:0]
	if cap(termBuf) < need {
		termBuf = make([]byte, 0, need)
	}
	buf := termBuf

	// \033[H = move cursor home, \033[J = clear to end of screen.
	// Termux requires this instead of cursor-home + overwrite to avoid tearing.
	buf = append(buf, "\033[H\033[J"...)

	// ── Render pixel pairs as half-block characters ─────────────────────────
	for y := 0; y < h; y += 2 {
		lastBG, lastFG := -2, -2 // sentinel: force emit on first pixel

		for x := 0; x < w; x++ {
			valTop := displayData[y*w+x]
			valBot := -1.0
			if y+1 < h {
				valBot = displayData[(y+1)*w+x]
			}

			// Map smooth iteration value → LUT index via cosine-smooth cycling.
			idxTop := -1
			if valTop >= 0 {
				idxTop = valToIdx(valTop, colorScale, lutSizeF)
			}
			idxBot := -1
			if valBot >= 0 {
				idxBot = valToIdx(valBot, colorScale, lutSizeF)
			}

			// Emit ANSI only when colour changes (run-length compression).
			if idxTop != lastBG {
				if idxTop < 0 {
					buf = append(buf, bgBlack...)
				} else {
					buf = append(buf, lut.BG[idxTop]...)
				}
				lastBG = idxTop
			}
			if idxBot != lastFG {
				if idxBot < 0 {
					buf = append(buf, fgBlack...)
				} else {
					buf = append(buf, lut.FG[idxBot]...)
				}
				lastFG = idxBot
			}

			buf = append(buf, blockChar...)
		}
		buf = append(buf, resetSeq...)
		buf = append(buf, '\n')
	}

	// ── Status bar ──────────────────────────────────────────────────────────
	zoomExp := zoom.MantExp(nil)
	log10Zoom := float64(zoomExp) * 0.30103
	cxF, _ := cx.Float64()
	cyF, _ := cy.Float64()

	modeStr := "Mandelbrot"
	if juliaMode {
		modeStr = fmt.Sprintf("Julia(%.4f%+.4fi)", juliaR, juliaI)
	}

	flags := ""
	if histoEQ {
		flags += " [EQ]"
	}
	if adaptIter {
		flags += " [AI]"
	}

	buf = append(buf, fmt.Sprintf(
		"%s | Pos:%.2e%+.2ei | Z:10^%.1f | It:%d | %s | D:%.2f%s | h:help",
		modeStr, cxF, cyF, log10Zoom, maxIter, currentPaletteName, colorDensity, flags,
	)...)

	// ── Help overlay ────────────────────────────────────────────────────────
	if showHelp {
		buf = append(buf, []byte(`
┌─ MOVEMENT ──────────────────────────────────────┐
│ w/a/s/d    Move          z/x    Zoom ×1.5/÷1.5  │
│ Z/X        Zoom ×10/÷10  r      Reset view       │
├─ QUALITY ───────────────────────────────────────┤
│ i/o        Iters ×2/÷2   I/O    Iters ×8/÷8     │
│ A          Toggle auto-iter scaling              │
│ e          Toggle histogram equalization         │
├─ COLOR ─────────────────────────────────────────┤
│ c          Cycle palette  n      Custom hex pal  │
│ k/l        Density ×1.2  K/L    Density ×5/÷5   │
├─ MODES ─────────────────────────────────────────┤
│ J          Toggle Julia mode                     │
│ ,/.        Julia real ∓0.01  ;/'  Julia imag ∓  │
├─ NAVIGATION ────────────────────────────────────┤
│ v          Find Minibrot  (q to abort mid-search)│
│ g          I Feel Lucky   (random deep zoom)     │
│ 1-9        Load bookmark  !-( (S+1-9) Save       │
├─ EXPORT ────────────────────────────────────────┤
│ P          1920×1080 PPM  p      Current-size PNG│
│ M          Zoom-out MP4   F      Zoom-in MP4     │
├─ OTHER ─────────────────────────────────────────┤
│ h          Toggle help    q      Quit            │
└─────────────────────────────────────────────────┘`)...)
	}

	// Retain grown capacity so next frame reuses it without allocation.
	termBuf = buf
	os.Stdout.Write(buf)
}

