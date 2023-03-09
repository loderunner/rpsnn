import * as React from 'react'
import { useCallback, useEffect, useMemo, useState } from 'react'

import type { RPSNetwork } from 'rps-network'

enum Choice {
  ROCK = 0,
  PAPER = 1,
  SCISSORS = 2,
}

enum Outcome {
  Player = 'Player',
  Computer = 'Computer',
  Draw = 'Draw',
}

function winsOver(choice: Choice): Choice {
  switch (choice) {
    case Choice.ROCK:
      return Choice.PAPER
    case Choice.PAPER:
      return Choice.SCISSORS
    case Choice.SCISSORS:
      return Choice.ROCK
  }
}

function choiceEmoji(choice: Choice): string {
  switch (choice) {
    case Choice.ROCK:
      return '👊'
    case Choice.PAPER:
      return '✋'
    case Choice.SCISSORS:
      return '✌️'
  }
}

function play(network: RPSNetwork, playerChoice: Choice): Choice {
  let computerChoice = Choice.ROCK
  const probs = network.probs()

  for (let i = 1; i < 3; i++) {
    if (probs[i] > probs[computerChoice]) {
      computerChoice = i
    }
  }

  const input = new Float32Array(3).fill(0)
  input[playerChoice] = 1

  network.backward(input, winsOver(playerChoice), 0.1)
  network.forward(input)

  return computerChoice
}

class Game {
  constructor(public playerChoice: Choice, public computerChoice: Choice) {}

  public get outcome(): Outcome {
    if (winsOver(this.playerChoice) === this.computerChoice) {
      return Outcome.Computer
    }
    if (winsOver(this.computerChoice) === this.playerChoice) {
      return Outcome.Player
    }
    return Outcome.Draw
  }
}

export default function App() {
  const [games, setGames] = useState<Game[]>([])
  const [network, setNetwork] = useState<RPSNetwork>()
  const [showProbs, setShowProbs] = useState(false)

  useEffect(() => {
    ;(async () => {
      const m = await import('rps-network')
      await m.default()
      setNetwork(new m.RPSNetwork(3, 8, 3))
    })()
  }, [])

  const toggleProbs = useCallback(() => {
    setShowProbs((showProbs) => !showProbs)
  }, [])

  const playGame = useCallback(
    (playerChoice: Choice) => {
      const computerChoice = play(network!, playerChoice)
      setGames([...games, new Game(playerChoice, computerChoice)])
    },
    [games, network]
  )

  const probs = network?.probs()
  let probsTable
  if (probs) {
    probsTable = (
      <table>
        <thead>
          <tr>
            <th scope="col">{choiceEmoji(Choice.ROCK)}</th>
            <th scope="col">{choiceEmoji(Choice.PAPER)}</th>
            <th scope="col">{choiceEmoji(Choice.SCISSORS)}</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>{probs[Choice.ROCK].toFixed(2)}</td>
            <td>{probs[Choice.PAPER].toFixed(2)}</td>
            <td>{probs[Choice.SCISSORS].toFixed(2)}</td>
          </tr>
        </tbody>
      </table>
    )
  }

  const gameItems = useMemo(
    () =>
      games.map((g, i) => (
        <tr key={i}>
          <td className="text-center">{choiceEmoji(g.playerChoice)}</td>
          <td className="text-center">{choiceEmoji(g.computerChoice)}</td>
          <td>{g.outcome}</td>
        </tr>
      )),
    [games]
  )

  return (
    <div>
      <div>
        <button onClick={() => playGame(Choice.ROCK)}>
          {choiceEmoji(Choice.ROCK)}
        </button>
        <button onClick={() => playGame(Choice.PAPER)}>
          {choiceEmoji(Choice.PAPER)}
        </button>
        <button onClick={() => playGame(Choice.SCISSORS)}>
          {choiceEmoji(Choice.SCISSORS)}
        </button>
      </div>
      <div>
        <button onClick={toggleProbs}>Show probs</button>
        <p>{showProbs ? probsTable : null}</p>
      </div>
      <table className="table-auto">
        <thead>
          <tr>
            <th scope="col">Player</th>
            <th scope="col">Computer</th>
            <th scope="col">Outcome</th>
          </tr>
        </thead>
        <tbody>{gameItems}</tbody>
      </table>
    </div>
  )
}
