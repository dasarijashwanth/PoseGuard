import { describe, expect, it } from "vitest";
import {
  classifyCameraError,
  constraintFallbacks,
  preferredDeviceId,
} from "./constraints";

function fakeDevices(): MediaDeviceInfo[] {
  return [
    { kind: "videoinput", label: "USB Webcam", deviceId: "usb-1" },
    { kind: "videoinput", label: "FaceTime HD Camera", deviceId: "front-1" },
    { kind: "audioinput", label: "Mic", deviceId: "mic-1" },
  ] as MediaDeviceInfo[];
}

describe("preferredDeviceId", () => {
  it("prefers a front-facing camera by label", () => {
    expect(preferredDeviceId(fakeDevices())).toBe("front-1");
  });

  it("falls back to the first video input when no front label matches", () => {
    const devices = [
      { kind: "videoinput", label: "USB Webcam", deviceId: "usb-1" },
    ] as MediaDeviceInfo[];
    expect(preferredDeviceId(devices)).toBe("usb-1");
  });

  it("returns undefined when there are no video inputs", () => {
    expect(preferredDeviceId([])).toBeUndefined();
  });

  it("returns undefined when labels are empty (pre-permission)", () => {
    const devices = [
      { kind: "videoinput", label: "", deviceId: "" },
    ] as MediaDeviceInfo[];
    expect(preferredDeviceId(devices)).toBeUndefined();
  });
});

describe("constraintFallbacks", () => {
  it("returns three attempts from best to most permissive", () => {
    const attempts = constraintFallbacks("cam-1");
    expect(attempts).toHaveLength(3);
    expect(attempts.map((a) => a.label)).toEqual(["hd", "sd", "any"]);
  });

  it("pins the known deviceId on the first two attempts", () => {
    const [hd, sd, any] = constraintFallbacks("cam-1");
    expect(JSON.stringify(hd.constraints)).toContain("cam-1");
    expect(JSON.stringify(sd.constraints)).toContain("cam-1");
    expect(JSON.stringify(any.constraints)).not.toContain("cam-1");
  });

  it("uses facingMode when no deviceId is known", () => {
    const [hd] = constraintFallbacks();
    expect(JSON.stringify(hd.constraints)).toContain("user");
    expect(JSON.stringify(hd.constraints)).not.toContain("deviceId");
  });
});

describe("classifyCameraError", () => {
  it("maps NotAllowedError to permission-denied with guidance", () => {
    const info = classifyCameraError("NotAllowedError");
    expect(info.code).toBe("permission-denied");
    expect(info.guidance).toMatch(/address bar/i);
  });

  it("maps NotFoundError to no-device", () => {
    expect(classifyCameraError("NotFoundError").code).toBe("no-device");
  });

  it("maps OverconstrainedError to overconstrained", () => {
    expect(classifyCameraError("OverconstrainedError").code).toBe(
      "overconstrained",
    );
  });

  it("maps NotReadableError to in-use", () => {
    expect(classifyCameraError("NotReadableError").code).toBe("in-use");
  });

  it("maps unknown names to unknown with a title and guidance", () => {
    const info = classifyCameraError("SomethingWeird");
    expect(info.code).toBe("unknown");
    expect(info.title.length).toBeGreaterThan(0);
    expect(info.guidance.length).toBeGreaterThan(0);
  });
});
