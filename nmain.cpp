#include <memory>
#include <utility>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

std::random_device rd;
std::mt19937 gen(rd());



class Gene {
protected:
    double fitness;
public:
    virtual ~Gene() = default;
    virtual void calculateFitness() = 0;
    virtual void mutate() = 0;
    //virtual void mutate(double mutationRate) = 0;
    virtual std::unique_ptr<Gene> clone() const = 0;

    virtual std::pair<std::unique_ptr<Gene>, std::unique_ptr<Gene>>
        crossover(const Gene& other) const = 0;
    double getFitness() const { return fitness; }
};


class BitGene : public Gene {
    private:
        std::vector<bool> alleles;
        float mutationRate;
    public:
        BitGene(int n) { initialize(n); }
        void initialize(int n) {
            std::vector<bool> alleles;
            std::uniform_int_distribution<> dis(0, 1);
            for (int i = 0; i < n; i++) {
                alleles.push_back(dis(gen));
            }
            this->alleles = alleles;
            this->mutationRate = 0.01;
        }

        void calculateFitness() override {
            int fitness = 0;
            std::for_each(this->alleles.begin(), this->alleles.end(), [&] (int n) {
                fitness += n;
            });
            this->fitness = fitness;
        }

        std::unique_ptr<Gene> clone() const override {
            auto clone = std::make_unique<BitGene>(this->alleles.size());
            for (int i = 0; i < alleles.size(); i++) {
                clone->setAllele(i, this->alleles[i]);
            }
            clone->fitness = this->fitness;

            return clone;
        }

        //void mutate(double mutationRate) override {
            //std::uniform_real_distribution<> dis(0.0, 1.0);
            //for (int i = 0; i < this->alleles.size(); i++){
                //if (dis(gen) < mutationRate) {
                    //this->alleles[i] = !this->alleles[i];
                //}
            //}
        //}

        void mutate() override {
            std::uniform_real_distribution<> dis(0.0, 1.0);
            for (int i = 0; i < this->alleles.size(); i++){
                if (dis(gen) < this->mutationRate) {
                    this->alleles[i] = !this->alleles[i];
                }
            }
        }

        void setAllele(int idx, bool val) {
            this->alleles[idx] = val;
        }

        bool getAllele(int idx) const {
            return this->alleles[idx];
        }

        std::pair<std::unique_ptr<Gene>, std::unique_ptr<Gene>>
            crossover(const Gene& other) const override {
                const BitGene& otherBit = dynamic_cast<const BitGene&>(other);
                auto offspring1 = std::make_unique<BitGene>(this->alleles.size());
                auto offspring2 = std::make_unique<BitGene>(this->alleles.size());

                std::uniform_int_distribution<int> dis(1, this->alleles.size() - 1);
                int r = dis(gen);

                for(int i = 0; i < this->alleles.size(); i++) {
                    if (i < r) {
                        offspring1->setAllele(i, this->alleles[i]);
                        offspring2->setAllele(i, otherBit.getAllele(i));
                    } else {
                        offspring1->setAllele(i, otherBit.getAllele(i));
                        offspring2->setAllele(i, this->alleles[i]);
                    }
                }
                return { std::move(offspring1), std::move(offspring2) };
            }
};


class Population {
    private:
        std::vector<std::unique_ptr<Gene>> individuals;
        std::unique_ptr<Gene> createGene(int n) const {
            return std::make_unique<BitGene>(n);
        }
    public:
        Population(int size, int n) {
            for (int i = 0; i < size; i++) {
                auto gene = createGene(n);
                individuals.push_back(std::move(gene));
            }
        }

        std::pair<const Gene*, const Gene*> uniformParentSelection() const {
            std::uniform_int_distribution<>dis(0, individuals.size() - 1);
            return {individuals[dis(gen)].get(), individuals[dis(gen)].get()};
        }

        void evolve() {
            std::vector<std::unique_ptr<Gene>> newPopulation;

            while (newPopulation.size() < individuals.size()) {
                auto [parent1, parent2] = uniformParentSelection();
                auto [offspring1, offspring2] = parent1->crossover(*parent2);
                offspring1->mutate();
                offspring2->mutate();
                offspring1->calculateFitness();
                offspring2->calculateFitness();
                newPopulation.push_back(std::move(offspring1));
                if(newPopulation.size() < individuals.size()) {
                    newPopulation.push_back(std::move(offspring2));
                }
            }

            individuals = std::move(newPopulation);
            std::sort(individuals.begin(), individuals.end(),
                    [](const std::unique_ptr<Gene>& a, const std::unique_ptr<Gene>& b) {
                        return a->getFitness() > b->getFitness();
                        });
        }

        const Gene& getBestIndividual() const {
            return *std::max_element(individuals.begin(), individuals.end(),
                    [](const std::unique_ptr<Gene>& a, const std::unique_ptr<Gene>& b) {
                        return a->getFitness() < b->getFitness();
                        })->get();
        }

        double getAverageFitness() const {
            double sum = 0.0;
            for (const auto& individual : individuals) {
                sum += individual->getFitness();
            }
            return sum / individuals.size();
        }
};


int main(void) {
    Population pop(10, 5);
    for (int generation = 0; generation < 100; generation++) {
        pop.evolve();
        //if (generation % 100 == 0) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            //std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
        //}
    }
    return 0;
}



