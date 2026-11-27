/* Switch between the saved test-performance images. */
(() => {
  "use strict";
  const chart = document.getElementById("comparison-chart");
  const buttons = document.querySelectorAll("[data-metric]");
  const metrics = {
    accuracy: [
      "accuracy",
      "Accuracy is the share of all predictions that are correct.",
    ],
    precision: [
      "precision",
      "Precision is the share of predicted rises that are actual rises.",
    ],
    recall: [
      "recall",
      "Recall is the share of actual rises the model correctly identifies.",
    ],
  };
  if (!chart) return;

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      const metric = button.dataset.metric;
      if (!Object.hasOwn(metrics, metric)) return;
      chart.src = `assets/images/comparison-${metric}.png`;
      chart.alt = button.dataset.alt;
      document.getElementById("comparison-title").textContent =
        `Test ${metrics[metric][0]}`;
      document.getElementById("metric-description").textContent =
        metrics[metric][1] + " The dashed line marks the persistence baseline.";
      buttons.forEach((other) =>
        other.setAttribute("aria-pressed", String(other === button)),
      );
    });
  });

  chart.addEventListener("load", () => {
    chart.width = chart.naturalWidth;
    chart.height = chart.naturalHeight;
  });
  document.documentElement.classList.add("js");
})();
