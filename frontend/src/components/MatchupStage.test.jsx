import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import MatchupStage from "./MatchupStage.jsx";
import { calculateDamage } from "../api/championsInsightApi.js";

vi.mock("../api/championsInsightApi.js", () => ({
  calculateDamage: vi.fn(),
  resolveApiUrl: vi.fn((pathOrUrl = "") => pathOrUrl),
}));

const userPokemon = {
  name: "Dragonite",
  ability: "Inner Focus",
  item: "",
  types: ["Dragon", "Flying"],
  image: "/dragonite.png",
  stats: {
    hp: 198,
    attack: 138,
    defense: 119,
    specialAttack: 158,
    specialDefense: 121,
    speed: 105,
  },
  moves: ["Dragon Claw"],
};

const opponentPokemon = {
  name: "Weavile",
  ability: "Pickpocket",
  item: "",
  types: ["Dark", "Ice"],
  image: "/weavile.png",
  stats: {
    hp: 145,
    attack: 140,
    defense: 85,
    specialAttack: 65,
    specialDefense: 105,
    speed: 145,
  },
  moves: ["Knock Off"],
  abilities: ["Pickpocket"],
};

describe("MatchupStage status controls", () => {
  afterEach(() => {
    cleanup();
  });

  beforeEach(() => {
    calculateDamage.mockResolvedValue({
      attacker: "Dragonite",
      defender: "Weavile",
      speed: { result: "Weavile moves first." },
      damage: {
        move: "Dragon Claw",
        category: "physical",
        moveType: "dragon",
        percentRange: "30% - 35%",
        koChance: "Possible 3HKO",
        notes: [],
        modifiers: { type: 1 },
      },
    });
  });

  it("sends mutually exclusive burn and paralysis statuses to damage calculation", async () => {
    const user = userEvent.setup();
    render(
      <MatchupStage
        onOpponentUpdate={vi.fn()}
        onSelectMove={vi.fn()}
        onUserFormChange={vi.fn()}
        opponentPokemon={opponentPokemon}
        selectedMove="Dragon Claw"
        selectedMoveSource="user"
        userPokemon={userPokemon}
      />
    );

    const userStatusGroup = screen.getByRole("group", {
      name: "Your Pokemon status",
    });
    const opponentStatusGroup = screen.getByRole("group", {
      name: "Opponent Pokemon status",
    });

    await user.click(within(userStatusGroup).getByLabelText("Burn"));
    await waitFor(() => {
      expect(calculateDamage.mock.calls.at(-1)?.[0].attacker.status).toBe("burn");
    });

    await user.click(within(userStatusGroup).getByLabelText("Para"));
    await waitFor(() => {
      const payload = calculateDamage.mock.calls.at(-1)?.[0];
      expect(payload.attacker.status).toBe("paralysis");
      expect(within(userStatusGroup).getByLabelText("Burn")).not.toBeChecked();
      expect(within(userStatusGroup).getByLabelText("Para")).toBeChecked();
    });

    await user.click(within(opponentStatusGroup).getByLabelText("Burn"));
    await waitFor(() => {
      expect(calculateDamage.mock.calls.at(-1)?.[0].defender.status).toBe("burn");
    });
  });

  it("sends expanded field effects to damage calculation", async () => {
    const user = userEvent.setup();
    render(
      <MatchupStage
        onOpponentUpdate={vi.fn()}
        onSelectMove={vi.fn()}
        onUserFormChange={vi.fn()}
        opponentPokemon={opponentPokemon}
        selectedMove="Dragon Claw"
        selectedMoveSource="user"
        userPokemon={userPokemon}
      />
    );

    await user.click(screen.getAllByLabelText("Helping Hand")[0]);
    await user.click(screen.getAllByLabelText("Friend Guard")[1]);
    await user.click(screen.getByLabelText("Gravity"));

    await waitFor(() => {
      const payload = calculateDamage.mock.calls.at(-1)?.[0];
      expect(payload.field.gravity).toBe(true);
      expect(payload.field.attackerSide.helpingHand).toBe(true);
      expect(payload.field.defenderSide.friendGuard).toBe(true);
    });
  });

  it("shows conditional ability toggles and forwards their state", async () => {
    const user = userEvent.setup();
    render(
      <MatchupStage
        onOpponentUpdate={vi.fn()}
        onSelectMove={vi.fn()}
        onUserFormChange={vi.fn()}
        opponentPokemon={{ ...opponentPokemon, ability: "Multiscale" }}
        selectedMove="Dragon Claw"
        selectedMoveSource="user"
        userPokemon={{ ...userPokemon, ability: "Flash Fire" }}
      />
    );

    expect(screen.getByLabelText("Flash Fire active")).toBeChecked();
    expect(screen.getByLabelText("Multiscale active")).toBeChecked();

    await user.click(screen.getByLabelText("Flash Fire active"));
    await user.click(screen.getByLabelText("Multiscale active"));

    await waitFor(() => {
      const payload = calculateDamage.mock.calls.at(-1)?.[0];
      expect(payload.attacker.abilityOn).toBeUndefined();
      expect(payload.defender.currentHp).toBe(144);
    });
  });
});
