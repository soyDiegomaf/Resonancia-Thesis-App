import { useEffect, useState } from "react";
import { getModels, runSegmentation, getSlice, getDownloadUrl } from "./api";

const loadingSteps = [
  "Loading selected model checkpoint",
  "Preparing MRI modalities",
  "Handling missing modalities",
  "Running full-volume inference",
  "Generating segmentation overlay",
  "Finalizing results",
];

function FileInput({ label, name, onChange, optional = false }) {
  return (
    <label className="file-card">
      <div>
        <span>{label}</span>
        <small>{optional ? "Optional" : "Required modality"}</small>
      </div>
      <input
        type="file"
        accept=".nii,.gz,.nii.gz"
        onChange={(e) => onChange(name, e.target.files[0])}
      />
    </label>
  );
}

function ResultImage({ title, src }) {
  if (!src) return null;

  return (
    <div className="viewer-card">
      <h3>{title}</h3>
      <img className="result-image" src={src} alt={title} />
    </div>
  );
}

function LoadingDots() {
  return (
    <span className="loading-dots">
      <span>.</span>
      <span>.</span>
      <span>.</span>
    </span>
  );
}

function ProcessPanel({ activeStep }) {
  return (
    <div className="process-panel">
      <h3>Processing Study</h3>

      <div className="process-list">
        {loadingSteps.map((step, index) => (
          <div
            className={
              index < activeStep
                ? "process-step done"
                : index === activeStep
                  ? "process-step active"
                  : "process-step"
            }
            key={step}
          >
            <span className="process-dot" />
            <p>{step}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [files, setFiles] = useState({
    t1: null,
    t1ce: null,
    t2: null,
    flair: null,
    groundTruth: null,
  });

  const [models, setModels] = useState({});
  const [modelName, setModelName] = useState("segresnet");
  const [sliceIndex, setSliceIndex] = useState("");
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [showNotice, setShowNotice] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [isUpdatingSlice, setIsUpdatingSlice] = useState(false);

  useEffect(() => {
    async function loadModels() {
      try {
        const data = await getModels();
        setModels(data);

        if (Object.keys(data).length > 0) {
          setModelName(Object.keys(data)[0]);
        }
      } catch (err) {
        setError(err.message);
      }
    }

    loadModels();
  }, []);

  useEffect(() => {
    if (!isRunning) return;

    setActiveStep(0);

    const interval = setInterval(() => {
      setActiveStep((prev) =>
        prev >= loadingSteps.length - 1 ? loadingSteps.length - 1 : prev + 1,
      );
    }, 1200);

    return () => clearInterval(interval);
  }, [isRunning]);

  function handleFileChange(name, file) {
    setFiles((prev) => ({
      ...prev,
      [name]: file || null,
    }));
  }

  async function handleSubmit(e) {
    e.preventDefault();

    setStatus("Running inference");
    setError("");
    setResult(null);
    setIsRunning(true);

    try {
      const data = await runSegmentation(files, sliceIndex, modelName);
      setResult(data);
      setSliceIndex(String(data.slice_index));
      setStatus("Segmentation complete.");
    } catch (err) {
      setError(err.message);
      setStatus("");
    } finally {
      setIsRunning(false);
    }
  }

  async function handleSliderChange(value) {
    setSliceIndex(value);

    if (!result?.session_id) return;

    setIsUpdatingSlice(true);

    try {
      const data = await getSlice(result.session_id, value);

      setResult((prev) => ({
        ...prev,
        slice_index: data.slice_index,
        images: data.images,
      }));
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUpdatingSlice(false);
    }
  }

  const selectedModel = models[modelName];

  return (
    <>
      <header className="site-header">
        <div className="header-inner">
          <h1>Resonancia</h1>
          <p className="header-description">
            Missing-modality brain tumor segmentation from multi-modal MRI.
          </p>
        </div>
      </header>

      <main className="app">
        {showNotice && (
          <section className="notice">
            <p>
              This application is a research prototype for brain tumor
              segmentation and is not intended for clinical diagnosis or
              treatment planning.
            </p>

            <button
              type="button"
              className="notice-close"
              onClick={() => setShowNotice(false)}
            >
              ×
            </button>
          </section>
        )}

        <section className="section-block">
          <div className="section-heading">
            <h2>Upload MRI Modalities</h2>
            <p>
              Upload any available BRaTS-style MRI sequences. Missing modalities
              will be handled automatically by the selected model workflow.
            </p>
          </div>

          <form onSubmit={handleSubmit}>
            <div className="model-section">
              <label className="model-title">Model</label>

              <div className="model-row">
                <div className="model-select">
                  <select
                    value={modelName}
                    onChange={(e) => setModelName(e.target.value)}
                  >
                    {Object.entries(models).map(([key, model]) => (
                      <option value={key} key={key}>
                        {model.display_name}
                      </option>
                    ))}
                  </select>
                </div>

                {selectedModel && (
                  <div className="model-info">
                    <p>
                      <strong>Architecture:</strong>{" "}
                      {selectedModel.architecture}
                    </p>
                    <p>
                      <strong>Checkpoint:</strong> {selectedModel.checkpoint}
                    </p>
                    <p>
                      <strong>Input:</strong> {selectedModel.input_channels}{" "}
                      channels
                    </p>
                    <p>
                      <strong>Output:</strong> {selectedModel.output_channels}{" "}
                      channels
                    </p>
                    <p>
                      <strong>Missing modality:</strong>{" "}
                      {selectedModel.missing_modality_strategy}
                    </p>
                  </div>
                )}
              </div>
            </div>

            <div className="file-grid">
              <FileInput label="T1 MRI" name="t1" onChange={handleFileChange} />
              <FileInput
                label="T1CE MRI"
                name="t1ce"
                onChange={handleFileChange}
              />
              <FileInput label="T2 MRI" name="t2" onChange={handleFileChange} />
              <FileInput
                label="FLAIR MRI"
                name="flair"
                onChange={handleFileChange}
              />
              <FileInput
                label="Ground Truth Mask"
                name="groundTruth"
                onChange={handleFileChange}
                optional
              />
            </div>

            <div className="actions-row">
              <div className="slice-control">
                <label>Initial slice index</label>

                <div className="slice-selector">
                  <button
                    type="button"
                    className="slice-btn"
                    onClick={() =>
                      setSliceIndex((prev) =>
                        prev === "" ? 0 : Math.max(0, Number(prev) - 1),
                      )
                    }
                  >
                    −
                  </button>

                  <input
                    type="number"
                    value={sliceIndex}
                    placeholder="Middle"
                    onChange={(e) => setSliceIndex(e.target.value)}
                  />

                  <button
                    type="button"
                    className="slice-btn"
                    onClick={() =>
                      setSliceIndex((prev) =>
                        prev === "" ? 1 : Number(prev) + 1,
                      )
                    }
                  >
                    +
                  </button>
                </div>
              </div>

              <button type="submit" disabled={isRunning}>
                {isRunning ? (
                  <>
                    Running <LoadingDots />
                  </>
                ) : (
                  "Run Segmentation"
                )}
              </button>
            </div>

            {status && (
              <p className={isRunning ? "status running" : "status"}>
                {status}
                {isRunning && <LoadingDots />}
              </p>
            )}

            {error && <p className="error">{error}</p>}
          </form>

          {isRunning && <ProcessPanel activeStep={activeStep} />}
        </section>

        {result && (
          <section className="section-block results-block">
            <div className="section-heading results-heading-row">
              <div>
                <h2>Segmentation Results</h2>
                <p>
                  Showing slice {result.slice_index} of {result.depth - 1}.
                  {isUpdatingSlice && " Updating slice..."}
                </p>
              </div>

              <a
                className="download-btn"
                href={getDownloadUrl(result.session_id)}
                download
              >
                Download Mask
              </a>
            </div>

            <div className="summary-grid">
              <div>
                <strong>Model</strong>
                <span>{result.model?.display_name}</span>
              </div>

              <div>
                <strong>Slice</strong>
                <span>
                  {result.slice_index} / {result.depth - 1}
                </span>
              </div>

              <div>
                <strong>Tumor Voxels</strong>
                <span>{result.tumor_voxels}</span>
              </div>

              <div>
                <strong>Tumor Volume</strong>
                <span>{result.tumor_volume_cm3} cm³</span>
              </div>

              <div>
                <strong>Mean Tumor Probability</strong>
                <span>
                  {result.mean_tumor_probability === null
                    ? "No tumor predicted"
                    : `${(result.mean_tumor_probability * 100).toFixed(2)}%`}
                </span>
              </div>

              <div>
                <strong>Max Tumor Probability</strong>
                <span>{(result.max_tumor_probability * 100).toFixed(2)}%</span>
              </div>

              <div>
                <strong>Dice Score</strong>
                <span>
                  {result.dice === null ? "N/A" : result.dice.toFixed(4)}
                </span>
              </div>
            </div>

            {result.dice === null && (
              <p className="metric-note">
                Dice score requires an uploaded ground truth segmentation mask.
              </p>
            )}

            <div className="slice-slider-panel">
              <label>
                Slice viewer
                <span>
                  {result.slice_index} / {result.depth - 1}
                </span>
              </label>

              <input
                type="range"
                min="0"
                max={result.depth - 1}
                value={sliceIndex}
                onChange={(e) => handleSliderChange(e.target.value)}
              />
            </div>

            <div className="availability">
              <h3>Detected Modalities</h3>
              <div>
                {Object.entries(result.availability).map(
                  ([modality, available]) => (
                    <span
                      className={available ? "available" : "missing"}
                      key={modality}
                    >
                      {modality.toUpperCase()}:{" "}
                      {available ? "Available" : "Missing"}
                    </span>
                  ),
                )}
              </div>
            </div>

            <div className="viewer-grid">
              <ResultImage title="MRI Slice" src={result.images?.mri} />
              <ResultImage
                title="Segmentation Overlay"
                src={result.images?.overlay}
              />
              <ResultImage title="Predicted Mask" src={result.images?.mask} />
              <ResultImage
                title="Tumor Probability Map"
                src={result.images?.probability}
              />
            </div>
          </section>
        )}
      </main>

      <footer className="site-footer">
        <p>Resonancia · Missing-Modality Brain Tumor Segmentation</p>
        <p>Developed by DiegoMAF</p>
      </footer>
    </>
  );
}
