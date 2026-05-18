// Defines the single-page Champions Insight app shell.
import React, { useEffect, useState } from "react";
import ArchetypeDashboardPage from "./pages/ArchetypeDashboardPage.jsx";
import BattlePrepPage from "./pages/BattlePrepPage.jsx";
import CameraCapturePage from "./pages/CameraCapturePage.jsx";
import PokedexPage from "./pages/PokedexPage.jsx";

function App() {
  const [view, setView] = useState(() => {
    if (window.location.pathname === "/archetypes") {
      return "archetypes";
    }

    return "battle";
  });
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

  useEffect(() => {
    function handlePopState() {
      const nextPath = window.location.pathname;
      setRoutePath(nextPath);

      if (nextPath === "/archetypes") {
        setView("archetypes");
        return;
      }

      setView("battle");
    }

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  function navigateToPath(pathname, nextView) {
    window.history.pushState({}, "", pathname);
    setRoutePath(pathname);
    setView(nextView);
  }

  function navigateToBattle(search = "") {
    navigateToPath(`/${search}`, "battle");
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

  if (view === "archetypes" || routePath === "/archetypes") {
    return <ArchetypeDashboardPage onBack={() => navigateToBattle()} />;
  }

  return (
    <BattlePrepPage
      initialTeam={builtTeam}
      onOpenArchetypeDashboard={() => navigateToPath("/archetypes", "archetypes")}
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
