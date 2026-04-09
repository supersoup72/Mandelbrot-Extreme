// globals.go — Shared types, variables, and constants for the Mandelbrot renderer.
// Every other file in this package reads from or writes to these.
package main

import "math/big"

// ─────────────────────────────────────────────
//  Types
// ─────────────────────────────────────────────

// Color is an RGB triplet stored as uint8 for compact packing.
type Color struct{ R, G, B uint8 }

// PaletteLUT holds a pre-interpolated color table and its pre-built ANSI
// escape byte slices so we never call fmt.Sprintf inside the render loop.
type PaletteLUT struct {
	Colors []Color
	FG, BG [][]byte // foreground / background ANSI escape sequences
}

// Bookmark stores a named view position with full big.Float precision.
type Bookmark struct {
	Name    string
	Cx, Cy  string // serialised as big.Float text so precision survives round-trips
	ZoomExp int    // binary exponent of zoom value (zoom ≈ 2^ZoomExp)
	MaxIter int
}

// ─────────────────────────────────────────────
//  View state
// ─────────────────────────────────────────────

var (
	cx, cy, zoom *big.Float
	maxIter      int     = 500
	colorDensity float64 = 2.0
)

// ─────────────────────────────────────────────
//  Feature flags
// ─────────────────────────────────────────────

var (
	showHelp  bool
	juliaMode bool
	juliaR    float64 = -0.7
	juliaI    float64 = 0.27015
	histoEQ   bool    // histogram equalization on/off
	orbitTrap bool    // orbit-trap coloring (reserved for future use)
	adaptIter bool    = true // auto-scale maxIter with zoom depth
)

// ─────────────────────────────────────────────
//  Palette globals
// ─────────────────────────────────────────────

var palettes = map[string][]Color{
	// Blue/Gold: deep navy → royal blue → ice white → rich gold → midnight.
	// More stops and richer midpoints than before for a wider colour arc.
	"Blue/Gold": {
		{0, 2, 30}, {0, 20, 100}, {10, 80, 200},
		{80, 180, 255}, {240, 255, 255}, {255, 220, 80},
		{255, 140, 0}, {180, 60, 0}, {10, 5, 20},
	},
	// Fire: true black-body radiation curve — black→deep red→orange→yellow→white.
	"Fire": {
		{0, 0, 0}, {80, 0, 0}, {200, 30, 0},
		{255, 100, 0}, {255, 210, 0}, {255, 255, 160}, {255, 255, 255},
	},
	// Grayscale: pure luminance ramp, perceptually linear via gamma correction.
	"Grayscale": {{0, 0, 0}, {128, 128, 128}, {255, 255, 255}},
	// Neon: vivid cyberpunk — black→violet→magenta→cyan→lime→white.
	"Neon": {
		{0, 0, 0}, {60, 0, 120}, {200, 0, 255},
		{0, 200, 255}, {0, 255, 120}, {200, 255, 0}, {255, 255, 255},
	},
	// Ocean: midnight black → deep navy → tropical teal → foam white → coral.
	"Ocean": {
		{0, 0, 15}, {0, 20, 80}, {0, 80, 160},
		{0, 180, 200}, {80, 230, 220}, {240, 255, 255},
		{255, 200, 120}, {200, 100, 40},
	},
	// Inferno: matplotlib-style perceptually uniform dark→purple→red→yellow.
	"Inferno": {
		{0, 0, 4}, {20, 5, 60}, {80, 10, 120},
		{160, 30, 100}, {220, 80, 50}, {250, 160, 20},
		{252, 230, 100}, {255, 255, 200},
	},
	// Ultra: the classic UltraFractal "ultra" palette, more saturated stops.
	"Ultra": {
		{0, 0, 0}, {80, 20, 5}, {30, 5, 40}, {10, 2, 70}, {3, 5, 100},
		{0, 10, 140}, {10, 55, 180}, {20, 100, 220}, {60, 148, 230},
		{140, 195, 245}, {215, 240, 252}, {245, 238, 200},
		{252, 210, 100}, {255, 175, 0}, {210, 130, 0}, {140, 75, 0},
	},
	// Sunset: dusk palette — deep purple → crimson → amber → pale gold.
	"Sunset": {
		{5, 0, 25}, {50, 0, 80}, {140, 10, 60},
		{220, 50, 20}, {255, 130, 0}, {255, 210, 80}, {255, 245, 200},
	},
	// Candy: pastel rainbow — every hue at high lightness, very smooth arcs.
	"Candy": {
		{255, 180, 200}, {255, 200, 120}, {200, 255, 150},
		{120, 220, 255}, {180, 150, 255}, {255, 160, 220},
	},
	// Ice: cold whites and blues, minimal saturation — great for deep zooms.
	"Ice": {
		{0, 0, 20}, {10, 30, 80}, {30, 90, 160},
		{100, 180, 230}, {200, 230, 255}, {240, 248, 255}, {255, 255, 255},
	},
}

var currentPaletteName = "Blue/Gold"
var paletteKeys = []string{
	"Blue/Gold", "Fire", "Grayscale", "Neon", "Ocean",
	"Inferno", "Ultra", "Sunset", "Candy", "Ice",
}

var luts map[string]*PaletteLUT

const lutSize = 4096

// Pre-built constant ANSI sequences used when a pixel is pure black (inside set).
var (
	fgBlack  = []byte("\033[38;2;0;0;0m")
	bgBlack  = []byte("\033[48;2;0;0;0m")
	resetSeq = []byte("\033[0m")
)

// ─────────────────────────────────────────────
//  Render buffers (allocated once, reused)
// ─────────────────────────────────────────────

// renderData is the main display buffer — filled by renderMandelbrot each frame.
var renderData []float64
var renderDataW, renderDataH int

// scratchData is a separate buffer used by the minibrot finder / iFeelLucky
// so low-res search renders never clobber the main display buffer.
var scratchData []float64
var scratchW, scratchH int

// termBuf is the terminal output buffer. Capacity grows and is retained across
// frames to avoid GC pressure (zero-allocation rendering).
var termBuf []byte

// blockChar is the UTF-8 encoding of ▄ (LOWER HALF BLOCK, U+2584).
// Used to pack two pixel rows into one terminal character row.
var blockChar = []byte{0xe2, 0x96, 0x84}

// ─────────────────────────────────────────────
//  Concurrency / search
// ─────────────────────────────────────────────

// minibrotAbort is atomically set to 1 by the abort listener goroutine when
// the user presses 'q' during autoFindMinibrot or iFeelLucky.
var minibrotAbort int32

// ─────────────────────────────────────────────
//  Bookmarks
// ─────────────────────────────────────────────

// bookmarks holds 10 save slots (accessed via keys '1'-'9' and shift+'1'-'9').
var bookmarks [10]Bookmark

