import React, { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [blueprint, setBlueprint] = useState(null);
  const [referenceBom, setReferenceBom] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    if (!blueprint) {
      setError("Please select a blueprint image.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();

      formData.append("blueprint", blueprint);

      if (referenceBom) {
        formData.append("reference_bom", referenceBom);
      }

      const response = await fetch(
        `${API_URL}/analyze-blueprint`,
        {
          method: "POST",
          body: formData,
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          errorText || "Analysis failed."
        );
      }

      const data = await response.json();

      setResult(data);

    } catch (err) {
      setError(
        err.message || "Unable to connect to backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setBlueprint(null);
    setReferenceBom(null);
    setResult(null);
    setError("");

    document
      .getElementById("blueprint-input")
      .value = "";

    document
      .getElementById("reference-input")
      .value = "";
  };

  return (
    <div className="app">

      <header className="header">
        <div>
          <h1>BOM Mismatch Detection System</h1>
          <p>
            Deep Learning-Based Engineering Blueprint Analysis
          </p>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          Backend Connected
        </div>
      </header>

      <main className="container">

        <section className="upload-section">

          <div className="upload-card">

            <h2>Upload Blueprint</h2>

            <p className="description">
              Upload an engineering blueprint image
              containing a Bill of Materials.
            </p>

            <label
              htmlFor="blueprint-input"
              className="file-box"
            >
              <span className="upload-icon">📄</span>

              <strong>
                {blueprint
                  ? blueprint.name
                  : "Choose blueprint image"}
              </strong>

              <span>
                PNG, JPG or JPEG
              </span>
            </label>

            <input
              id="blueprint-input"
              type="file"
              accept=".png,.jpg,.jpeg"
              onChange={(e) =>
                setBlueprint(
                  e.target.files[0] || null
                )
              }
            />

          </div>


          <div className="upload-card">

            <h2>Reference BOM</h2>

            <p className="description">
              Optional. Upload the expected BOM
              as a CSV file for mismatch detection.
            </p>

            <label
              htmlFor="reference-input"
              className="file-box"
            >
              <span className="upload-icon">📊</span>

              <strong>
                {referenceBom
                  ? referenceBom.name
                  : "Choose reference CSV"}
              </strong>

              <span>
                CSV format
              </span>
            </label>

            <input
              id="reference-input"
              type="file"
              accept=".csv"
              onChange={(e) =>
                setReferenceBom(
                  e.target.files[0] || null
                )
              }
            />

          </div>

        </section>


        <section className="action-section">

          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={loading}
          >
            {loading
              ? "Analyzing Blueprint..."
              : "Analyze Blueprint"}
          </button>

          <button
            className="reset-button"
            onClick={handleReset}
            disabled={loading}
          >
            Reset
          </button>

        </section>


        {error && (
          <div className="error-box">
            <strong>Error:</strong> {error}
          </div>
        )}


        {loading && (
          <div className="loading-box">
            <div className="spinner"></div>

            <div>
              <strong>
                Processing blueprint...
              </strong>

              <p>
                OCR and LayoutLMv3 are extracting
                BOM information.
              </p>
            </div>
          </div>
        )}


        {result && (
          <Results result={result} />
        )}

      </main>

    </div>
  );
}


function Results({ result }) {

  const bom = result.extracted_bom || [];

  const comparison =
    result.comparison;

  return (
    <section className="results">

      <div className="results-header">

        <div>
          <h2>Analysis Results</h2>

          <p>
            {result.filename}
          </p>
        </div>

        <div className="result-badge">
          {result.comparison_available
            ? comparison.status
            : "BOM EXTRACTED"}
        </div>

      </div>


      <div className="stats">

        <div className="stat-card">
          <span>OCR Detections</span>
          <strong>
            {result.ocr_detections}
          </strong>
        </div>

        <div className="stat-card">
          <span>OCR Rows</span>
          <strong>
            {result.ocr_rows}
          </strong>
        </div>

        <div className="stat-card">
          <span>BOM Rows</span>
          <strong>
            {result.extracted_bom_rows}
          </strong>
        </div>

        <div className="stat-card">
          <span>Reference Rows</span>
          <strong>
            {result.reference_bom_rows}
          </strong>
        </div>

      </div>


      <div className="table-section">

        <h3>Extracted BOM</h3>

        {bom.length === 0 ? (
          <p>No BOM rows were extracted.</p>
        ) : (
          <div className="table-wrapper">

            <table>

              <thead>
                <tr>
                  <th>Part No</th>
                  <th>Description</th>
                  <th>Material</th>
                  <th>UOM</th>
                  <th>Qty</th>
                </tr>
              </thead>

              <tbody>

                {bom.map((row, index) => (
                  <tr key={index}>

                    <td>
                      {row.PART_NO || "—"}
                    </td>

                    <td>
                      {row.DESCRIPTION || "—"}
                    </td>

                    <td>
                      {row.MATERIAL || "—"}
                    </td>

                    <td>
                      {row.UOM || "—"}
                    </td>

                    <td>
                      {row.QTY || "—"}
                    </td>

                  </tr>
                ))}

              </tbody>

            </table>

          </div>
        )}

      </div>


      {comparison && (
        <ComparisonReport comparison={comparison} />
      )}

    </section>
  );
}


function ComparisonReport({ comparison }) {

  return (
    <div className="comparison-section">

      <div className="comparison-header">

        <h3>
          BOM Comparison
        </h3>

        <div
          className={
            comparison.status === "MATCH"
              ? "match-badge"
              : "mismatch-badge"
          }
        >
          {comparison.status}
        </div>

      </div>


      <div className="comparison-summary">

        <div>
          <span>
            Fields Checked
          </span>

          <strong>
            {comparison.summary.total_fields_checked}
          </strong>
        </div>

        <div>
          <span>
            Matching
          </span>

          <strong>
            {comparison.summary.matching_fields}
          </strong>
        </div>

        <div>
          <span>
            Mismatched
          </span>

          <strong>
            {comparison.summary.mismatched_fields}
          </strong>
        </div>

      </div>


      <div className="table-wrapper">

        <table>

          <thead>

            <tr>
              <th>Part No</th>
              <th>Field</th>
              <th>Expected</th>
              <th>Detected</th>
              <th>Status</th>
            </tr>

          </thead>

          <tbody>

            {comparison.results.map(
              (item, index) => (

                <tr key={index}>

                  <td>
                    {item.PART_NO}
                  </td>

                  <td>
                    {item.FIELD}
                  </td>

                  <td>
                    {item.EXPECTED}
                  </td>

                  <td>
                    {item.DETECTED}
                  </td>

                  <td>

                    <span
                      className={
                        item.STATUS === "MATCH"
                          ? "match-text"
                          : "mismatch-text"
                      }
                    >
                      {item.STATUS}
                    </span>

                  </td>

                </tr>

              )
            )}

          </tbody>

        </table>

      </div>

    </div>
  );
}


export default App;