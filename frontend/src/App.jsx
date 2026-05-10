// Defines the single-page Champions Insight app shell.
import React, { useEffect, useState } from "react";
import { saveUserTeam } from "./api/championsInsightApi.js";
import BattlePrepPage from "./pages/BattlePrepPage.jsx";
import TeamBuilderPage from "./pages/TeamBuilderPage.jsx";
import CameraCapturePage from "./pages/CameraCapturePage.jsx";
import PokedexPage from "./pages/PokedexPage.jsx";

function App() {
  const [view, setView] = useState("battle");
  const [builtTeam, setBuiltTeam] = useState(null);
  const [theme, setTheme] = useState(() => {
    const savedTheme = window.localStorage.getItem("champions-insight-theme");
    if (savedTheme === "light" || savedTheme === "dark") {
      return savedTheme;
    }

    return window.matchMedia?.("(prefers-color-scheme: light)")?.matches
      ? "light"
      : "dark";
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("champions-insight-theme", theme);
  }, [theme]);

  async function saveBuiltTeam(team) {
    const savedTeam = await saveUserTeam(team);
    setBuiltTeam(savedTeam.team);
    setView("battle");
  }

  // Phone-only guided camera page.
  if (window.location.pathname === "/camera") {
    return <CameraCapturePage />;
  }

  if (view === "team-builder") {
    return (
      <TeamBuilderPage
        initialTeam={builtTeam}
        onCancel={() => setView("battle")}
        onSave={saveBuiltTeam}
      />
    );
  }

  if (view === "pokedex") {
    return <PokedexPage onBack={() => setView("battle")} />;
  }

  return (
    <BattlePrepPage
      initialTeam={builtTeam}
      onOpenTeamBuilder={() => setView("team-builder")}
      onOpenPokedex={() => setView("pokedex")}
      theme={theme}
      onToggleTheme={() => setTheme((currentTheme) => (
        currentTheme === "dark" ? "light" : "dark"
      ))}
    />
  );
}

export default App;
