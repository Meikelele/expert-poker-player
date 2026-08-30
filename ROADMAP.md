# Branch: feat/dqn-agent

## Cel

Zaimplementować agenta Deep Q-Network dla Ultimate Texas Hold'em.

DQN ma korzystać z istniejących abstrakcji projektu:

- StateEncoder
- RewardFunction
- UTHObservation
- StepResult
- UTHGame

Implementacja ma obsługiwać oba warianty reprezentacji stanu:

- raw, 373 elementy
- features, 393 elementy

oraz oba warianty funkcji nagrody:

- net_profit
- stake_scaled_net_profit

DQN nie może korzystać z:

- RoundState
- kart krupiera przed showdownem
- kart spalonych
- kolejności kart w talii
- przyszłego wyniku rozdania

---

# Założenia algorytmu

DQN aproksymuje funkcję:

Q(s, a)

dla sześciu akcji zdefiniowanych przez środowisko UTH.

Sieć otrzymuje wektor stanu generowany przez StateEncoder i zwraca
jedną wartość Q dla każdej wartości Action.

Liczba wyjść sieci:

ACTION_COUNT = 6

Akcje nielegalne w aktualnej fazie muszą być maskowane.

Maskowanie jest wymagane w dwóch miejscach:

1. podczas wyboru akcji przez agenta
2. podczas obliczania max Q dla stanu następnego w celu Bellmana

Agent nigdy nie może wybrać nielegalnej akcji.

---

# Bellman target

Dla przejścia terminalnego:

y = r

Dla przejścia nieterminalnego:

y = r + gamma \* max Q_target(s', a')

gdzie maksimum jest obliczane wyłącznie po legalnych akcjach w s'.

Sieć target jest oddzielona od sieci policy.

---

# Eksploracja

Podczas treningu używana jest strategia epsilon-greedy.

Z prawdopodobieństwem epsilon:

- wybierana jest losowa legalna akcja

Z prawdopodobieństwem 1 - epsilon:

- wybierana jest legalna akcja o największej wartości Q

Epsilon maleje podczas treningu zgodnie z ustalonym harmonogramem.

Losowanie eksploracyjne ma korzystać z jawnie seedowanego generatora RNG.

---

# Replay buffer

Przejścia treningowe mają być przechowywane w replay bufferze.

Transition powinien zawierać co najmniej:

- state
- action index
- reward
- next_state
- terminated
- next legal action mask

Dla przejścia terminalnego:

- next_state może być None
- nie wykonuje się bootstrapu z target network

ReplayBuffer:

- ma ograniczoną pojemność
- nadpisuje najstarsze próbki
- losuje mini-batche
- wykorzystuje własny seedowany RNG
- nie zależy od UTHGame

---

# Sieć neuronowa

Implementacja w PyTorch.

QNetwork:

input:
StateEncoder.output_size

hidden layers:
konfigurowalne

activation:
ReLU

output:
len(Action)

Domyślna architektura początkowa:

input
|
Linear
|
ReLU
|
Linear
|
ReLU
|
Linear -> 6 Q-values

Domyślne warstwy ukryte:

256
256

Nie dodawać na tym etapie:

- CNN
- RNN
- LSTM
- Transformer
- dueling DQN
- double DQN
- prioritized replay
- noisy networks

Najpierw klasyczny DQN jako kontrolowany wariant badawczy.

---

# Konfiguracja treningu

DQNConfig powinien przechowywać jawne hiperparametry.

Minimum:

- learning_rate
- gamma
- batch_size
- replay_capacity
- warmup_steps
- target_sync_interval
- epsilon_start
- epsilon_end
- epsilon_decay_steps
- hidden_sizes
- training_episodes
- seed

Wartości domyślne mają być rozsądne dla pilota, ale wszystkie istotne
parametry powinny dać się zapisać razem z wynikiem eksperymentu.

Konfiguracja ma być walidowana.

---

# Reproducibility

Seed treningu musi kontrolować co najmniej:

- inicjalizację PyTorch
- epsilon-greedy RNG
- replay sampling RNG
- generowanie rozdań treningowych

Nie należy korzystać z jednego globalnego RNG dla wszystkich elementów.

Dwa uruchomienia z tą samą konfiguracją i seedem powinny być
reprodukowane na tym samym środowisku wykonawczym w zakresie
oczekiwanym dla używanych operacji PyTorch.

---

# Training pipeline

Schemat:

UTHObservation
|
StateEncoder
|
state vector
|
DQNAgent
|
legal epsilon-greedy action
|
UTHGame.step(action)
|
StepResult
|
RewardFunction
|
reward
|
Transition
|
ReplayBuffer
|
mini-batch
|
policy network
target network
|
Bellman target
|
loss
|
optimizer step

---

# Evaluation

Model po treningu ma być możliwy do wykorzystania jako zwykły Agent.

Podczas ewaluacji:

epsilon = 0

Agent powinien działać przez istniejący interfejs:

select_action(
observation: UTHObservation
) -> Action

Dzięki temu będzie można wykorzystać istniejącą infrastrukturę
run_simulation oraz metrics bez tworzenia specjalnego środowiska
ewaluacyjnego dla DQN.

---

# Checkpoint

Checkpoint powinien umożliwiać późniejszą ewaluację wytrenowanego modelu.

Minimum:

- policy network state_dict
- input size
- hidden sizes
- StateRepresentation
- RewardType
- training seed
- DQNConfig lub jego serializowalne parametry

Checkpoint nie powinien serializować całego obiektu UTHGame.

---

# Commit plan

## 1. feat(dqn): add q network

Zakres:

- dodać zależność PyTorch
- utworzyć pakiet dqn
- QNetwork
- konfigurowalne hidden sizes
- input size
- output size = len(Action)
- podstawowe testy forward pass
- test wymiaru dla State A
- test wymiaru dla State B

Nie implementować jeszcze agenta ani treningu.

---

## 2. feat(dqn): add legal action masking

Zakres:

- stabilne mapowanie Action <-> index
- funkcja tworząca legal action mask
- funkcja maskująca Q-values
- nielegalne akcje nie mogą wygrać argmax
- terminal observation nie jest stanem decyzyjnym
- testy dla PREFLOP, FLOP i RIVER

Maskowanie musi być możliwe do wykorzystania zarówno przez agenta,
jak i przy Bellman target.

---

## 3. feat(dqn): add seeded epsilon greedy agent

Zakres:

- DQNAgent zgodny z Agent Protocol
- StateEncoder jako zależność
- QNetwork jako zależność
- epsilon
- własny random.Random
- greedy selection
- random legal action
- żadnych nielegalnych akcji
- epsilon = 0 daje greedy policy
- epsilon = 1 daje eksplorację tylko po legalnych akcjach
- terminal observation odrzucane

---

## 4. feat(dqn): add replay buffer

Zakres:

- Transition
- ReplayBuffer
- capacity
- FIFO overwrite
- seeded sampling
- batch sampling
- obsługa terminalnych transition
- next_state
- next legal action mask
- test reprodukowalności samplowania

ReplayBuffer nie może zależeć od UTHGame ani od konkretnego rewardu.

---

## 5. feat(dqn): add training configuration

Zakres:

- DQNConfig
- walidacja hiperparametrów
- epsilon schedule
- stabilna konfiguracja serializowalna do wyników eksperymentu

Minimum:

learning_rate
gamma
batch_size
replay_capacity
warmup_steps
target_sync_interval
epsilon_start
epsilon_end
epsilon_decay_steps
hidden_sizes
training_episodes
seed

Testy:

- poprawne wartości
- błędne zakresy
- epsilon start
- epsilon end
- monotoniczny decay

---

## 6. feat(dqn): add bellman target computation

Zakres:

- obliczanie targetu DQN
- terminal target = reward
- non-terminal target = reward + gamma \* legal max target Q
- target network
- next-action masking
- brak gradientu przez target network

Testy powinny używać kontrolowanych Q-values i sprawdzać dokładne wyniki.

To jest krytyczny commit metodologiczny.

---

## 7. feat(dqn): add optimization step

Zakres:

- mini-batch -> tensors
- policy Q dla wykonanej akcji
- Bellman target
- loss
- optimizer
- backward
- optimizer.step
- target network synchronization

Domyślny loss:

SmoothL1Loss

Domyślny optimizer:

Adam

Target network synchronizowany okresowo przez skopiowanie policy state_dict.

Testy:

- parametry policy network zmieniają się po update
- target network nie zmienia się przed synchronizacją
- target network zgadza się z policy po synchronizacji

---

## 8. feat(dqn): add training loop

Zakres:

- prawdziwy UTHGame
- StateEncoder
- RewardFunction
- DQNAgent
- ReplayBuffer
- epsilon schedule
- optimization steps
- target synchronization
- wiele epizodów
- terminalne rozdania

Każde przejście powinno mieć postać:

s, a, r, s', done, next legal mask

Trening nie może korzystać z RoundState.

Zebrać podstawowe statystyki treningu:

- episode reward
- loss
- epsilon
- liczba kroków
- liczba optimizer updates

---

## 9. feat(dqn): add model checkpoints

Zakres:

- zapis policy network
- odczyt policy network
- zapis architektury
- zapis reprezentacji stanu
- zapis funkcji nagrody
- zapis config
- zapis seed

Po odczycie checkpointu agent ma produkować te same Q-values
dla tego samego wejścia co przed zapisem.

---

## 10. test(dqn): verify deterministic training pipeline

Test integracyjny.

Zweryfikować:

- UTHGame -> encoder -> DQN -> reward -> replay -> update
- oba StateEncoder warianty są kompatybilne
- oba RewardFunction warianty są kompatybilne
- agent nie korzysta z RoundState
- nielegalne akcje nigdy nie są wybierane
- target nie bootstrapuje po terminal state
- identyczny seed daje identyczny krótki przebieg treningowy

Nie wymagać identyczności pomiędzy różnymi systemami CPU/GPU,
jeżeli PyTorch nie gwarantuje bitowej deterministyczności.

---

## 11. experiments: run dqn pilot

Dodać mały eksperyment pilotowy.

Cel pilota:

- sprawdzić czy trening faktycznie działa
- sprawdzić czy loss pozostaje skończony
- sprawdzić czy Q-values nie eksplodują
- oszacować czas treningu
- sprawdzić zapis checkpointu
- sprawdzić możliwość późniejszej ewaluacji

Pilot nie jest finalnym eksperymentem pracy.

Nie wyciągać na jego podstawie wniosków o przewadze modelu.

Zapisać:

- konfigurację
- seed
- state representation
- reward type
- liczbę epizodów
- podstawowe statystyki treningu
- czas wykonania

---

## 12. docs: document DQN method

Rozszerzyć:

thesis/chapters/05_metody_rl.tex

Opisać:

- funkcję Q
- przybliżenie siecią neuronową
- architekturę QNetwork
- sześć wyjść odpowiadających akcjom
- legal action masking
- epsilon-greedy
- replay buffer
- policy network
- target network
- Bellman target
- gamma
- Smooth L1 loss
- optimizer
- target synchronization
- konfigurację treningu
- reproducibility
- checkpointy
- sposób integracji z State A / State B
- sposób integracji z Reward A / Reward B

Rozróżnić:

- metodę
- hiperparametry pilota
- przyszłe hiperparametry finalnych eksperymentów

Nie przedstawiać pilota jako wyników końcowych.

---

# Definition of Done

Branch jest gotowy, gdy:

- PyTorch jest jawnie zależnością projektu
- QNetwork przyjmuje dowolny istniejący StateEncoder.output_size
- QNetwork zwraca dokładnie len(Action) Q-values
- istnieje stabilne mapowanie Action na indeks
- wszystkie wybory akcji respektują legal_actions
- Bellman target respektuje legal_actions stanu następnego
- DQNAgent spełnia Agent Protocol
- epsilon-greedy ma osobny seedowany RNG
- ReplayBuffer ma osobny seedowany RNG
- istnieje DQNConfig
- epsilon schedule jest deterministyczny
- istnieje policy network
- istnieje target network
- terminal transition nie wykonuje bootstrapu
- non-terminal transition używa gamma
- optimizer aktualizuje wyłącznie policy network
- target network jest synchronizowany jawnie
- training loop korzysta z UTHObservation, nie RoundState
- training loop korzysta z RewardFunction
- training loop korzysta z StateEncoder
- trening obsługuje State A i State B
- trening obsługuje Reward A i Reward B
- agent po treningu może być użyty przez istniejący evaluation runner
- checkpoint można zapisać i odtworzyć
- seed jest zapisywany razem z konfiguracją
- istnieją testy integracyjne całego pipeline
- wykonano pilot treningowy
- pełny pytest przechodzi
- dokumentacja pracy odpowiada implementacji

---

# Poza zakresem brancha

Nie implementować:

- Double DQN
- Dueling DQN
- Prioritized Experience Replay
- Rainbow
- n-step returns
- distributional DQN
- noisy networks
- PPO
- REINFORCE
- MCTS
- Q-Learning tabelaryczny
- reward shaping
- curriculum learning
- self-play
- GPU-specific optimization
- hyperparameter search
- finalnej macierzy eksperymentów

---

# Następny branch

feat/policy-gradient-agent
