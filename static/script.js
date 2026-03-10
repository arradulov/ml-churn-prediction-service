document.getElementById("predictBtn").addEventListener("click", async () => {
    const fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
        alert("Выберите файл!");
        return;
    }

    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    // Создаем таблицу
    let tableHTML = "<table class='table table-striped'><thead><tr>";
    Object.keys(data[0]).forEach(key => {
        tableHTML += `<th>${key}</th>`;
    });
    tableHTML += "</tr></thead><tbody>";

    data.forEach(row => {
        tableHTML += "<tr>";
        Object.entries(row).forEach(([key, value]) => {
            if (key === "churn_probability" && value > 0.5) {
                tableHTML += `<td style="color:red;font-weight:bold">${value.toFixed(2)}</td>`;
            } else {
                tableHTML += `<td>${value}</td>`;
            }
        });
        tableHTML += "</tr>";
    });

    tableHTML += "</tbody></table>";

    document.getElementById("resultTable").innerHTML = tableHTML;
});