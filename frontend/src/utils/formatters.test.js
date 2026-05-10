// Tests shared formatting helpers.
import { describe, expect, it } from "vitest";
import { formatBytes } from "./formatters.js";


describe("formatBytes", () => {
  it("formats byte values", () => {
    expect(formatBytes(67)).toBe("67 B");
  });

  it("formats kilobyte values", () => {
    expect(formatBytes(1536)).toBe("1.5 KB");
  });

  it("formats megabyte values", () => {
    expect(formatBytes(2 * 1024 * 1024)).toBe("2.0 MB");
  });

  it("handles invalid values", () => {
    expect(formatBytes(Number.NaN)).toBe("unknown size");
  });
});
