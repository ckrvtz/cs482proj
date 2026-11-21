//VERSION=3
// Cloud-masked per-band median composite over all acquisitions in the request's time range
// (M1-T05). Template: the two lines marked TEMPLATE are filled in from config/project.yaml by
// desertification.ingest.render_evalscript, and the SHA-256 of the filled-in script goes in
// each download's provenance sidecar.
//
// Output (UINT16, Decision log #45): one band per entry in BANDS holding the median surface
// reflectance x 10000 (harmonised DN; Sentinel Hub's default harmonizeValues=true makes values
// comparable across the 2022-01-25 processing-baseline change and clamps negatives to 0),
// then the count of valid observations per pixel. Pixels with no valid observation are 0 in
// every band (count = 0).
const BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]; // TEMPLATE
const INVALID_SCL = [0, 1, 2, 3, 8, 9, 10, 11]; // TEMPLATE

function setup() {
  return {
    input: [{ bands: BANDS.concat(["SCL", "dataMask"]), units: "DN" }],
    output: { bands: BANDS.length + 1, sampleType: "UINT16" },
    mosaicking: "ORBIT"
  };
}

function median(values) {
  if (values.length === 0) return 0;
  values.sort((a, b) => a - b);
  const mid = Math.floor(values.length / 2);
  return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

function evaluatePixel(samples) {
  const valid = samples.filter(s => s.dataMask === 1 && !INVALID_SCL.includes(s.SCL));
  const out = BANDS.map(b => median(valid.map(s => s[b])));
  out.push(valid.length);
  return out;
}
