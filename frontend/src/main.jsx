// Mounts the React application into the HTML root element.
import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import './styles.css';
import './styles/ChampionsInsightTheme.css';

createRoot(document.getElementById("root")).render(<App />);
