// Loads browser-style test matchers for Vitest.
import "@testing-library/jest-dom/vitest";
import React from "react";
import { vi } from "vitest";

globalThis.React = React;

Object.defineProperty(URL, "createObjectURL", {
  writable: true,
  value: vi.fn(() => "blob:champions-insight-test"),
});
