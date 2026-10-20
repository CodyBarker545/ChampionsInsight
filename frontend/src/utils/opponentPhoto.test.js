import { afterEach, describe, expect, it, vi } from "vitest";
import { prepareOpponentPhotoForUpload } from "./opponentPhoto.js";

afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

describe("opponent upload preparation", () => {
  it.each([32, 2 * 1024 * 1024])("uploads the original file unchanged (%i bytes)", async (size) => {
    const file = new File([new Uint8Array(size)], "phone.png", { type: "image/png" });
    const decode = vi.fn();
    vi.stubGlobal("createImageBitmap", decode);
    const encode = vi.spyOn(HTMLCanvasElement.prototype, "toBlob");
    expect(await prepareOpponentPhotoForUpload(file)).toBe(file);
    expect(decode).not.toHaveBeenCalled();
    expect(encode).not.toHaveBeenCalled();
  });
});
