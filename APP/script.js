document.getElementById("evaluateBtn").addEventListener("click", async () => {
    const age = document.getElementById("age").value;
    const fileInput = document.getElementById("cvFile");
    const file = fileInput.files[0];
    const resultDiv = document.getElementById("result");
    const loadingDiv = document.getElementById("loading");

    if (!age || !file) {
        alert("Please enter your age and select a CV file.");
        return;
    }
    
    loadingDiv.style.display = "block";
    resultDiv.classList.add("hidden");

    const formData = new FormData();
    formData.append("age", age);
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/evaluate_cv", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        const evalData = data.data.evaluation;

        const levelBadge = document.getElementById("levelBadge");
        const level = evalData.level || "Unknown";
        levelBadge.textContent = `Level: ${level}`;
        levelBadge.className = "level-badge";
        if (level === "High") levelBadge.classList.add("level-high");
        else if (level === "Medium") levelBadge.classList.add("level-medium");
        else levelBadge.classList.add("level-low");

        const scoreValue = evalData.confidence || 0;
        document.getElementById("scoreValue").textContent = `${scoreValue.toFixed(1)}`;
        document.getElementById("scoreBar").style.width = `${scoreValue}%`;

        const probChart = document.getElementById("probabilityChart");
        probChart.innerHTML = "";

        const scoreBreakdown = evalData.score_breakdown || {};
        for (const [label, percent] of Object.entries(scoreBreakdown)) {
            const pct = percent || 0;
            const color = getColorForLevel(label);
            probChart.innerHTML += `
                <div class="probability-item">
                    <div class="probability-label">
                        <span>${label}</span>
                        <span>${pct.toFixed(1)}%</span>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width:${pct}%; background:${color};"></div>
                    </div>
                </div>
            `;
        }

        resultDiv.classList.remove("hidden");
        loadingDiv.style.display = "none";

    } catch (err) {
        loadingDiv.style.display = "none";
        alert("Error: " + err.message);
        console.error(err);
    }
});

function getColorForLevel(level) {
    switch(level) {
        case "High": return "#2ecc71";
        case "Medium": return "#f39c12";
        case "Low": return "#e74c3c";
        default: return "#3498db";
    }
}
