const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function getModels() {
  const response = await fetch(`${API_BASE_URL}/models`);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Could not load models.");
  }

  return data;
}

export async function runSegmentation(files, sliceIndex, modelName) {
  const formData = new FormData();

  if (files.t1) formData.append("t1", files.t1);
  if (files.t1ce) formData.append("t1ce", files.t1ce);
  if (files.t2) formData.append("t2", files.t2);
  if (files.flair) formData.append("flair", files.flair);
  if (files.groundTruth) formData.append("ground_truth", files.groundTruth);
  if (sliceIndex !== "") formData.append("slice_index", sliceIndex);

  formData.append("model_name", modelName);

  const response = await fetch(`${API_BASE_URL}/segment`, {
    method: "POST",
    body: formData,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Segmentation failed.");
  }

  return data;
}

export async function getSlice(sessionId, sliceIndex) {
  const response = await fetch(`${API_BASE_URL}/slice/${sessionId}/${sliceIndex}`);
  const data = await response.json();

  if (!response.ok) {
    throw new Error(data.detail || "Slice update failed.");
  }

  return data;
}

export function getDownloadUrl(sessionId) {
  return `${API_BASE_URL}/download/${sessionId}`;
}