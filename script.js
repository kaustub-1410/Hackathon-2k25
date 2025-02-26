// Mobile Menu Toggle
const mobileMenuButton = document.getElementById("mobile-menu-button");
const mobileMenu = document.getElementById("mobile-menu");
mobileMenuButton.addEventListener("click", () => {
  mobileMenu.classList.toggle("hidden");
});

// Section Navigation Logic
function showSection(sectionId) {
  const sections = document.querySelectorAll("main > section");
  sections.forEach((section) => {
    if (section.id === sectionId) {
      section.classList.remove("hidden-section");
    } else {
      section.classList.add("hidden-section");
    }
  });
}

// Listen to hash changes to display the appropriate section
window.addEventListener("hashchange", () => {
  const hash = window.location.hash.substring(1) || "home";
  showSection(hash);
  if (hash === "analysis") {
    initializeChart();
  }
  if (hash === "patient-records") {
    renderPatients();
  }
});

// On initial load, check hash and show that section
document.addEventListener("DOMContentLoaded", () => {
  const initialHash = window.location.hash.substring(1) || "home";
  showSection(initialHash);
  if (initialHash === "analysis") {
    initializeChart();
  }
  if (initialHash === "patient-records") {
    renderPatients();
  }
});

// ECharts initialization for Analysis Results
function initializeChart() {
  const chartContainer = document.getElementById("resultChart");
  if (chartContainer) {
    const chart = echarts.init(chartContainer);
    const option = {
      animation: false,
      title: {
        text: "Analysis Metrics",
        left: "center",
        textStyle: { fontSize: 16, fontWeight: "bold" },
      },
      radar: {
        indicator: [
          { name: "Clump Thickness", max: 10 },
          { name: "Cell Size", max: 10 },
          { name: "Cell Shape", max: 10 },
          { name: "Adhesion", max: 10 },
          { name: "Nuclei", max: 10 },
        ],
        splitArea: {
          areaStyle: {
            color: [
              "rgba(0,123,255,0.1)",
              "rgba(0,123,255,0.2)",
              "rgba(0,123,255,0.3)",
              "rgba(0,123,255,0.4)",
              "rgba(0,123,255,0.5)",
            ],
          },
        },
      },
      series: [
        {
          type: "radar",
          data: [
            {
              value: [7, 8, 6, 5, 7],
              name: "Current Analysis",
              lineStyle: { color: "#007bff", width: 2 },
              areaStyle: { color: "rgba(0,123,255,0.2)" },
            },
          ],
        },
      ],
    };
    chart.setOption(option);
  }
}

/* -------------------------- */
/*   Patient Records CRUD     */
/* -------------------------- */
let patients = [
  {
    id: "P-20231205-001",
    name: "Jane Smith",
    age: 45,
    lastTest: "2023-11-30",
    status: "Benign",
  },
  {
    id: "P-20231205-002",
    name: "Emily Johnson",
    age: 50,
    lastTest: "2023-12-01",
    status: "Malignant",
  },
];

function renderPatients() {
  const tbody = document.getElementById("patientsTableBody");
  tbody.innerHTML = "";
  patients.forEach((patient, index) => {
    const row = document.createElement("tr");
    row.classList.add("hover:bg-gray-100", "transition", "duration-200");
    row.innerHTML = `
      <td class="px-6 py-4 whitespace-nowrap text-gray-900">${patient.id}</td>
      <td class="px-6 py-4 whitespace-nowrap text-gray-900">${patient.name}</td>
      <td class="px-6 py-4 whitespace-nowrap text-gray-900">${patient.age}</td>
      <td class="px-6 py-4 whitespace-nowrap text-gray-900">${patient.lastTest}</td>
      <td class="px-6 py-4 whitespace-nowrap text-gray-900">${patient.status}</td>
      <td class="px-6 py-4 whitespace-nowrap">
        <button class="edit-btn bg-yellow-400 text-white py-1 px-2 rounded mr-2" data-index="${index}">Edit</button>
        <button class="delete-btn bg-red-600 text-white py-1 px-2 rounded" data-index="${index}">Delete</button>
      </td>
    `;
    tbody.appendChild(row);
  });
}

// Add new patient record
document
  .getElementById("addPatientForm")
  .addEventListener("submit", function (e) {
    e.preventDefault();
    const newId = document.getElementById("newId").value;
    const newName = document.getElementById("newName").value;
    const newAge = document.getElementById("newAge").value;
    const newLastTest = document.getElementById("newLastTest").value;
    const newStatus = document.getElementById("newStatus").value;
    patients.push({
      id: newId,
      name: newName,
      age: parseInt(newAge),
      lastTest: newLastTest,
      status: newStatus,
    });
    renderPatients();
    this.reset();
  });

// Delegate edit and delete actions using event delegation on tbody
document
  .getElementById("patientsTableBody")
  .addEventListener("click", function (e) {
    if (e.target.classList.contains("delete-btn")) {
      const index = e.target.getAttribute("data-index");
      patients.splice(index, 1);
      renderPatients();
    } else if (e.target.classList.contains("edit-btn")) {
      const index = e.target.getAttribute("data-index");
      const row = e.target.closest("tr");
      const patient = patients[index];
      row.innerHTML = `
        <td class="px-6 py-4 whitespace-nowrap">
          <input type="text" class="border rounded px-2 py-1" value="${patient.id}" />
        </td>
        <td class="px-6 py-4 whitespace-nowrap">
          <input type="text" class="border rounded px-2 py-1" value="${patient.name}" />
        </td>
        <td class="px-6 py-4 whitespace-nowrap">
          <input type="number" class="border rounded px-2 py-1" value="${patient.age}" />
        </td>
        <td class="px-6 py-4 whitespace-nowrap">
          <input type="date" class="border rounded px-2 py-1" value="${patient.lastTest}" />
        </td>
        <td class="px-6 py-4 whitespace-nowrap">
          <select class="border rounded px-2 py-1">
            <option value="Benign" ${patient.status === "Benign" ? "selected" : ""}>Benign</option>
            <option value="Malignant" ${patient.status === "Malignant" ? "selected" : ""}>Malignant</option>
          </select>
        </td>
        <td class="px-6 py-4 whitespace-nowrap">
          <button class="save-btn bg-green-600 text-white py-1 px-2 rounded mr-2" data-index="${index}">Save</button>
          <button class="cancel-btn bg-gray-500 text-white py-1 px-2 rounded" data-index="${index}">Cancel</button>
        </td>
      `;
    } else if (e.target.classList.contains("save-btn")) {
      const index = e.target.getAttribute("data-index");
      const row = e.target.closest("tr");
      const inputs = row.querySelectorAll("input, select");
      const updatedPatient = {
        id: inputs[0].value,
        name: inputs[1].value,
        age: parseInt(inputs[2].value),
        lastTest: inputs[3].value,
        status: inputs[4].value,
      };
      patients[index] = updatedPatient;
      renderPatients();
    } else if (e.target.classList.contains("cancel-btn")) {
      renderPatients();
    }
  });

