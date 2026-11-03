import { expect, test } from "@playwright/test";

// Placeholder: real flows (calibration, monitoring, export, offline reload)
// land with the camera and pose modules on later days.
test("app loads and shows the title", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "PoseGuard" })).toBeVisible();
});
