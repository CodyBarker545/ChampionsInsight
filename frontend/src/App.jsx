// Defines the single-page Champions Insight app shell.
import React, { useEffect, useState } from "react";
import BattlePrepPage from "./pages/BattlePrepPage.jsx";
import CameraCapturePage from "./pages/CameraCapturePage.jsx";
import PokedexPage from "./pages/PokedexPage.jsx";

function App() {
  const [view, setView] = useState("battle");
  const [routePath, setRoutePath] = useState(() => window.location.pathname);
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

  function navigateToBattle(search = "") {
    const nextUrl = `/${search}`;
    window.history.pushState({}, "", nextUrl);
    setRoutePath("/");
    setView("battle");
  }

  // Phone-only guided camera page.
  if (routePath === "/camera") {
    return (
      <CameraCapturePage
        onClose={() => navigateToBattle()}
      />
    );
  }

  if (view === "pokedex") {
    return <PokedexPage onBack={() => setView("battle")} />;
  }

  return (
    <BattlePrepPage
      initialTeam={builtTeam}
      onOpenPokedex={() => setView("pokedex")}
      onTeamSaved={setBuiltTeam}
      theme={theme}
      onToggleTheme={() => setTheme((currentTheme) => (
        currentTheme === "dark" ? "light" : "dark"
      ))}
    />
  );
}

export default App;
