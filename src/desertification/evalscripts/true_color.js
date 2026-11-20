//VERSION=3
// True-colour preview (B04, B03, B02) for visual AOI checks (M1-T02). Not for analysis.
// Reflectance x 2.5 for display; sampleType AUTO maps 0..1 to 0..255 (8-bit, cheapest PU).
// dataMask is returned as a 4th band (alpha) and is not counted in PU.
function setup() {
  return {
    input: [{ bands: ["B04", "B03", "B02", "dataMask"] }],
    output: { bands: 4, sampleType: "AUTO" }
  };
}

function evaluatePixel(s) {
  return [2.5 * s.B04, 2.5 * s.B03, 2.5 * s.B02, s.dataMask];
}
