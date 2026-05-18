// Tests user team roster rendering and selection.
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import TeamForm from "./TeamForm.jsx";


describe("TeamForm", () => {
  const team = [{ id: "aegis", name: "Aegis", image: "", moves: ["Power Strike"], stats: { speed: 100 } }];

  afterEach(() => {
    cleanup();
  });

  it("renders team member fields", () => {
    render(<TeamForm selectedIndex={0} team={team} onSelectMember={vi.fn()} />);

    expect(screen.getByRole("button", { name: /Aegis/ })).toBeInTheDocument();
    expect(screen.getByText("My Team")).toBeInTheDocument();
  });

  it("calls select handler when a slot is clicked", async () => {
    const user = userEvent.setup();
    const onSelectMember = vi.fn();
    render(<TeamForm selectedIndex={0} team={team} onSelectMember={onSelectMember} />);

    await user.click(screen.getByRole("button", { name: /Aegis/ }));

    expect(onSelectMember).toHaveBeenCalledWith(0);
  });

  it("does not open the builder when a filled slot is clicked", async () => {
    const user = userEvent.setup();
    const onEditMember = vi.fn();
    render(
      <TeamForm
        selectedIndex={0}
        team={team}
        onEditMember={onEditMember}
        onSelectMember={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /Aegis/ }));

    expect(onEditMember).not.toHaveBeenCalled();
  });

  it("opens the builder when an empty slot is clicked", async () => {
    const user = userEvent.setup();
    const onEditMember = vi.fn();
    render(
      <TeamForm
        selectedIndex={0}
        team={[{ id: "empty", name: "", image: "" }]}
        onEditMember={onEditMember}
        onSelectMember={vi.fn()}
      />,
    );

    await user.click(screen.getByRole("button", { name: /Pokemon 1/ }));

    expect(onEditMember).toHaveBeenCalledWith(0);
  });
});
