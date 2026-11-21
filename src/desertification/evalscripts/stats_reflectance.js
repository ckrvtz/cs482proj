//VERSION=3
// Statistical API: cloud-masked surface reflectance (0-1) of a few bands, for time series
// over a small stable target (M1-T05 harmonisation check). Pixels with an invalid SCL class
// or no data are excluded through the dataMask output. The TEMPLATE lines are filled in
// from config by desertification.ingest.render_evalscript.
const BANDS = ["B02", "B04", "B08", "B11"]; // TEMPLATE
const INVALID_SCL = [0, 1, 2, 3, 8, 9, 10, 11]; // TEMPLATE

function setup() {
  return {
    input: [{ bands: BANDS.concat(["SCL", "dataMask"]) }],
    output: [
      { id: "reflectance", bands: BANDS, sampleType: "FLOAT32" },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function evaluatePixel(s) {
  const ok = s.dataMask === 1 && !INVALID_SCL.includes(s.SCL) ? 1 : 0;
  return { reflectance: BANDS.map(b => s[b]), dataMask: [ok] };
}
