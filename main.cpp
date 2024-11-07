#include <cstddef>
#include <vector>

class Gene {
    public:
        virtual ~Gene() = default;
        virtual double getFitness() const = 0;
        virtual int getAge() const;
        virtual void updateAge();
        virtual void mutate() = 0;
        virtual void crossover(const Gene& other) = 0;
    protected:
        double fitness;
        int age;
};

void Gene::updateAge() {
    age++;
}

int Gene::getAge() const {
    return age;
}

class BitGene : public Gene {
    public:
        double getFitness() const override;
        void mutate() override;
        void crossover(const Gene& other) override;
    private:
        std::vector<bool> alelles;
};


class SelectionStrategy {
    public:
        virtual ~SelectionStrategy() = default;
        virtual std::vector<Gene*> select(const std::vector<Gene*>& population) = 0;
};

class RouletteWheelSelection : public SelectionStrategy {
public:
    std::vector<Gene*> select(const std::vector<Gene*>& population) override {
        // Roulette wheel selection logic
        // Select individuals based on their relative fitness
    }
};


class MutationStrategy {
    public:
        virtual ~MutationStrategy() = default;
        virtual void mutate(Gene& gene) = 0;
};


class CrossoverStrategy {
    public:
        virtual ~CrossoverStrategy() = default;
        virtual std::vector<Gene*> crossover(const std::vector<Gene*>& parent) = 0;
};


class FlipBitMutation : public MutationStrategy {
    public:
        void mutate(Gene& gene) override {
            // Flip Bit for BitGene
        }
};

class Population {
    public:
        size_t populationSize;
        double mutationRate;
        SelectionStrategy* selectionStrategy;
        MutationStrategy* mutationStrategy;

        Population(size_t popSize, SelectionStrategy* selection, MutationStrategy* mutation)
            : populationSize(popSize), selectionStrategy(selection), mutationStrategy(mutation) {
                initialize();
        }

        void initialize();
        void selection();
        void crossover();
        void mutate();
        void evolve() {
            selection();
            crossover();
            mutate();
        }
    private:
        std::vector<Gene*> individuals;
};

int main(void) {
    Population pop(100, new RouletteWheelSelection(), new FlipBitMutation());

    for (int gen = 0; gen < 100; ++gen) {
        pop.evolve();
    }

    return 0;
}
