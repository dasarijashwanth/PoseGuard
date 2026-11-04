/**
 * Pure camera helpers: device selection, constraint fallbacks, and error
 * classification. No browser globals are touched here, so everything is
 * unit-testable in Node.
 */
import type {
  CameraErrorCode,
  CameraErrorInfo,
  ConstraintAttempt,
} from "./types";

/**
 * Pick the preferred camera device id from an enumerateDevices() list.
 * Prefers a front-facing/integrated camera by label; falls back to the
 * first video input. Returns undefined when no camera exists or when
 * labels are empty (pre-permission), in which case the caller should use
 * facingMode constraints instead of a deviceId.
 */
export function preferredDeviceId(
  devices: MediaDeviceInfo[],
  preferFront = true,
): string | undefined {
  const cameras = devices.filter((d) => d.kind === "videoinput");
  if (cameras.length === 0) {
    return undefined;
  }
  if (preferFront) {
    const front = cameras.find((d) =>
      /front|facetime|integrated|built-?in/i.test(d.label),
    );
    if (front && front.deviceId) {
      return front.deviceId;
    }
  }
  return cameras[0].deviceId || undefined;
}

/**
 * Ordered getUserMedia attempts, from best quality to most permissive.
 * When a deviceId is known it is pinned with `exact` on the first two
 * attempts; the last attempt accepts any camera.
 */
export function constraintFallbacks(deviceId?: string): ConstraintAttempt[] {
  const selector = deviceId
    ? { deviceId: { exact: deviceId } }
    : { facingMode: "user" as const };
  return [
    {
      label: "hd",
      constraints: {
        video: {
          ...selector,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      },
    },
    {
      label: "sd",
      constraints: {
        video: {
          ...selector,
          width: { ideal: 640 },
          height: { ideal: 480 },
        },
        audio: false,
      },
    },
    {
      label: "any",
      constraints: { video: true, audio: false },
    },
  ];
}

const ERROR_COPY: Record<CameraErrorCode, { title: string; guidance: string }> =
  {
    "permission-denied": {
      title: "Camera access was blocked",
      guidance:
        "Click the camera icon in the address bar and set camera access to Allow, then try again. If that does not help, check your operating system privacy settings for camera access.",
    },
    "no-device": {
      title: "No camera found",
      guidance:
        "PoseGuard could not find a camera on this device. Connect a webcam and try again.",
    },
    overconstrained: {
      title: "Camera settings not supported",
      guidance:
        "None of the camera configurations PoseGuard tried worked on this device. Try a different camera if one is available.",
    },
    "in-use": {
      title: "Camera is already in use",
      guidance:
        "Another app or browser tab is using the camera. Close it and try again.",
    },
    unknown: {
      title: "Could not start the camera",
      guidance:
        "Something unexpected went wrong while starting the camera. Reload the page and try again.",
    },
  };

/**
 * Map a getUserMedia DOMException name to a stable error code plus
 * plain-English guidance. Pure function of the error name.
 */
export function classifyCameraError(errorName: string): CameraErrorInfo {
  let code: CameraErrorCode;
  switch (errorName) {
    case "NotAllowedError":
    case "SecurityError":
      code = "permission-denied";
      break;
    case "NotFoundError":
    case "DevicesNotFoundError":
      code = "no-device";
      break;
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      code = "overconstrained";
      break;
    case "NotReadableError":
    case "TrackStartError":
    case "AbortError":
      code = "in-use";
      break;
    default:
      code = "unknown";
  }
  const { title, guidance } = ERROR_COPY[code];
  return { code, title, guidance };
}
