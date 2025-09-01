document.getElementById("uploadForm").addEventListener("submit", async function(e) {
  e.preventDefault();

  const input = document.getElementById("imageInput");
  const file = input.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  const loading = document.getElementById("loading");
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "";
  loading.style.display = "block";

  try {
    const response = await fetch("http://localhost:8000/upload-image", {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    loading.style.display = "none";

    data.results.forEach(imagePath => {
      const img = document.createElement("img");
      img.src = imagePath; // Should be a valid URL or static path served by backend
      resultsDiv.appendChild(img);
    });
  } catch (err) {
    console.error(err);
    loading.innerText = "Error searching for similar images.";
  }
});
