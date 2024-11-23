#include <limits>
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
    double mutationRate;
public:
    virtual ~Gene() = default;
    virtual void calculateFitness() = 0;
    virtual void mutate() = 0;
    virtual double calculateDistance(const Gene& other) const = 0;
    virtual std::unique_ptr<Gene> clone() const = 0;

    virtual std::pair<std::unique_ptr<Gene>, std::unique_ptr<Gene>>
        crossover(const Gene& other) const = 0;
    double getFitness() const { return fitness; }
    void setMutationRate(double mutationRate) { this->mutationRate = mutationRate; }
    double getMutationRate() const { return mutationRate; }

    virtual void print(std::ostream& os) const = 0;
    friend std::ostream& operator<<(std::ostream& os, const Gene& gene) {
        gene.print(os);
        return os;
    }
};


class BitGene : public Gene {
    private:
        std::vector<bool> alleles;
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
            calculateFitness();
        }

        void calculateFitness() override {
            double fitness = 0;
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

        double calculateDistance(const Gene& other) const override {
            // Jaccard Distance
            const BitGene& otherBit = dynamic_cast<const BitGene&>(other);
            int intersection_count = 0;
            int union_count = 0;

            for (int i = 0; i < this->alleles.size(); i++) {
                if (this->getAllele(i) == 1 || otherBit.getAllele(i) == 1) {
                    union_count++;
                } else if (this->getAllele(i) == 1 && otherBit.getAllele(i) == 1) {
                    intersection_count++;
                }
            }
            double jaccard_similarity = (union_count == 0) ? 0 : double(intersection_count) / union_count;
            return 1.0 - jaccard_similarity;
        }

    void print(std::ostream& os) const override {
        for (bool allele : this->alleles) {
            os << (allele ? '1' : '0');
        }
    }
};


class IntGene : public Gene {
    private:
        std::vector<int> alleles;
        int minAllele;
        int maxAllele;
    public:
        IntGene(int n, int minAllele, int maxAllele) { initialize(n, minAllele, maxAllele); }
        void initialize(int n, int minAllele, int maxAllele) {
            std::vector<int> alleles;
            std::uniform_int_distribution<> dis(minAllele, maxAllele);
            for (int i = 0; i < n; i++) {
                alleles.push_back(dis(gen));
            }
            this->alleles = alleles;
            this->minAllele = minAllele;
            this->maxAllele = maxAllele;
            this->mutationRate = 0.01;
            calculateFitness();
        }

        void calculateFitness() override {
            double fitness = 0;
            std::for_each(this->alleles.begin(), this->alleles.end(), [&] (int n) {
                fitness += n;
                });
            this->fitness = fitness;
        }

        std::unique_ptr<Gene> clone() const override {
            auto clone = std::make_unique<IntGene>(this->alleles.size(), this->minAllele, this->maxAllele);
            for (int i = 0; i < alleles.size(); i++) {
                clone->setAllele(i, this->alleles[i]);
            }
            clone->fitness = this->fitness;
            return clone;
        }

        void mutate() override {
            std::uniform_real_distribution<> dis(0.0, 1.0);
            std::uniform_int_distribution<> disI(this->minAllele, this->maxAllele);
            for (int i = 0; i < this->alleles.size(); i++) {
                if (dis(gen) < this->mutationRate) {
                    this->alleles[i] = disI(gen);
                }
            }
        }

        void setAllele(int idx, int val) {
            this->alleles[idx] = val;
        }

        int getAllele(int idx) const {
            return this->alleles[idx];
        }

        std::pair<std::unique_ptr<Gene>, std::unique_ptr<Gene>>
            crossover(const Gene& other) const override {
                const IntGene& otherInt = dynamic_cast<const IntGene&>(other);
                auto offspring1 = std::make_unique<IntGene>(this->alleles.size(), this->minAllele, this->maxAllele);
                auto offspring2 = std::make_unique<IntGene>(this->alleles.size(), this->minAllele, this->maxAllele);
                std::uniform_int_distribution<int> dis(1, this->alleles.size() - 1);
                int r = dis(gen);

                for(int i = 0; i < this->alleles.size(); i++) {
                    if (i < r) {
                        offspring1->setAllele(i, this->alleles[i]);
                        offspring2->setAllele(i, otherInt.getAllele(i));
                    } else {
                        offspring1->setAllele(i, otherInt.getAllele(i));
                        offspring2->setAllele(i, this->alleles[i]);
                    }
                }
                return { std::move(offspring1), std::move(offspring2) };
            }

        double calculateDistance(const Gene& other) const override {
            // Euclidean_distance
            const IntGene& otherInt = dynamic_cast<const IntGene&>(other);
            double sum = 0;

            for (int i = 0; i < this->alleles.size(); i++) {
                double diff = this->getAllele(i) - otherInt.getAllele(i);
                sum += diff * diff;
            }
            return sum;
        }

    void print(std::ostream& os) const override {
        for (int i = 0; i < this->alleles.size(); i++) {
            os << this->alleles[i];
            if (i < this->alleles.size() - 1) {
                os << ' ';
            }
        }
    }
};


class RealGene : public Gene {
    private:
        std::vector<double> alleles;
        double minAllele;
        double maxAllele;
        double lowerMutationValue;
        double upperMutationValue;
    public:
        RealGene(int n, double minAllele, double maxAllele) { initialize(n, minAllele, maxAllele); }
        void initialize(int n, double minAllele, double maxAllele, double lowerMutationValue = -1.0, double upperMutationValue = 1.0) {
            std::vector<double> alleles;
            std::uniform_real_distribution<> dis(minAllele, maxAllele);
            for (int i = 0; i < n; i++) {
                alleles.push_back(dis(gen));
            }
            this->alleles = alleles;
            this->minAllele = minAllele;
            this->maxAllele = maxAllele;
            this->mutationRate = 0.01;
            this->lowerMutationValue = lowerMutationValue;
            this->upperMutationValue = upperMutationValue;
            calculateFitness();
        }

        void calculateFitness() override {
            double fitness = 0;
            std::for_each(this->alleles.begin(), this->alleles.end(), [&] (int n) {
                fitness += n * n;
                });
            this->fitness = fitness;
        }

        std::unique_ptr<Gene> clone() const override {
            auto clone = std::make_unique<RealGene>(this->alleles.size(), this->minAllele, this->maxAllele);
            for (int i = 0; i < alleles.size(); i++) {
                clone->setAllele(i, this->alleles[i]);
            }
            clone->fitness = this->fitness;
            return clone;
        }

        void mutate() override {
            std::uniform_real_distribution<> dis(0.0, 1.0);
            std::uniform_real_distribution<> disR(this->lowerMutationValue, this->upperMutationValue);
            for (int i = 0; i < this->alleles.size(); i++) {
                if (dis(gen) < this->mutationRate) {
                    this->alleles[i] += disR(gen);
                }
            }
        }

        void setAllele(int idx, int val) {
            this->alleles[idx] = val;
        }

        double getAllele(int idx) const {
            return this->alleles[idx];
        }

        std::pair<std::unique_ptr<Gene>, std::unique_ptr<Gene>>
            crossover(const Gene& other) const override {
                const RealGene& otherReal = dynamic_cast<const RealGene&>(other);
                auto offspring1 = std::make_unique<RealGene>(this->alleles.size(), this->minAllele, this->maxAllele);
                auto offspring2 = std::make_unique<RealGene>(this->alleles.size(), this->minAllele, this->maxAllele);
                std::uniform_int_distribution<int> dis(1, this->alleles.size() - 1);
                int r = dis(gen);

                for(int i = 0; i < this->alleles.size(); i++) {
                    if (i < r) {
                        offspring1->setAllele(i, this->alleles[i]);
                        offspring2->setAllele(i, otherReal.getAllele(i));
                    } else {
                        offspring1->setAllele(i, otherReal.getAllele(i));
                        offspring2->setAllele(i, this->alleles[i]);
                    }
                }
                return { std::move(offspring1), std::move(offspring2) };
            }

        double calculateDistance(const Gene& other) const override {
            // Euclidean_distance
            const RealGene& otherReal = dynamic_cast<const RealGene&>(other);
            double sum = 0;

            for (int i = 0; i < this->alleles.size(); i++) {
                double diff = this->getAllele(i) - otherReal.getAllele(i);
                sum += diff * diff;
            }
            return sum;
        }

    void print(std::ostream& os) const override {
        for (int i = 0; i < this->alleles.size(); i++) {
            os << this->alleles[i];
            if (i < this->alleles.size() - 1) {
                os << ' ';
            }
        }
    }
};


class Population {
    private:
        std::vector<std::unique_ptr<Gene>> individuals;
        std::unique_ptr<Gene> createGene(int n) const {
            return std::make_unique<BitGene>(n);
        }
        std::unique_ptr<Gene> createIntGene(int n, int minAllele, int maxAllele) const {
            return std::make_unique<IntGene>(n, minAllele, maxAllele);
        }
        std::unique_ptr<Gene> createRealGene(int n, double minAllele, double maxAllele) const {
            return std::make_unique<RealGene>(n, minAllele, maxAllele);
        }
    public:
        Population(int size, int n) {
            for (int i = 0; i < size; i++) {
                auto gene = createGene(n);
                individuals.push_back(std::move(gene));
            }
        }

        Population(int size, int n, int minAllele, int maxAllele) {
            for (int i = 0; i < size; i++) {
                auto gene = createIntGene(n, minAllele, maxAllele);
                individuals.push_back(std::move(gene));
            }
        }

        Population(int size, int n, double minAllele, double maxAllele) {
            for (int i = 0; i < size; i++) {
                auto gene = createRealGene(n, minAllele, maxAllele);
                individuals.push_back(std::move(gene));
            }
        }

        std::pair<const Gene*, const Gene*> uniformParentSelection() const {
            std::uniform_int_distribution<>dis(0, individuals.size() - 1);
            return {individuals[dis(gen)].get(), individuals[dis(gen)].get()};
        }

        std::pair<const Gene*, const Gene*> fitnessProportionalSelection() const {
            double totalFitness = 0.0;
            double leastFitness = std::numeric_limits<double>::max();

            for (const auto& individual : individuals) {
                totalFitness += individual->getFitness();
                leastFitness = std::min(leastFitness, individual->getFitness());
            }
            totalFitness -= leastFitness * this->individuals.size();

            std::uniform_real_distribution<> dis(0.0, totalFitness);
            double rnd1 = dis(gen);
            double rnd2 = dis(gen);

            const Gene* parent1 = nullptr;
            const Gene* parent2 = nullptr;
            double accumulatedFitness = 0.0;
            for(const auto& individual : individuals) {
                accumulatedFitness += individual->getFitness();
                accumulatedFitness -= leastFitness;
                if (parent1 == nullptr && accumulatedFitness >= rnd1) {
                    parent1 = individual.get();
                }
                if (parent2 == nullptr && accumulatedFitness >= rnd2) {
                    parent2 = individual.get();
                }
                if (parent1 != nullptr && parent2 != nullptr) {
                    break;
                }
            }
            return {parent1, parent2};
        }



        const Gene& getBestIndividual() const {
            return *std::max_element(individuals.begin(), individuals.end(),
                    [](const std::unique_ptr<Gene>& a, const std::unique_ptr<Gene>& b) {
                        return a->getFitness() < b->getFitness();
                        })->get();
        }

        const Gene& getWorseIndividual() const {
            return *std::min_element(individuals.begin(), individuals.end(),
                    [](const std::unique_ptr<Gene>& a, const std::unique_ptr<Gene>& b) {
                        return a->getFitness() < b->getFitness();
                        })->get();
        }

        double calcualteDiveristy() const {
            // pairwise distance average
            double totalDistance = 0.0;
            int n = individuals.size() * (individuals.size() - 1) / 2;
            for (int i = 0; i < individuals.size(); i++) {
                for (int j = i + 1; j < individuals.size(); j++) {
                    totalDistance += individuals[i]->calculateDistance(*individuals[j]);
                }
            }
            return totalDistance / n;
        }

        double getAverageFitness() const {
            double sum = 0.0;
            for (const auto& individual : individuals) {
                sum += individual->getFitness();
            }
            return sum / individuals.size();
        }

        void sortPopulationWithFitness() {
            std::sort(individuals.begin(), individuals.end(),
                    [](const std::unique_ptr<Gene>& a, const std::unique_ptr<Gene>& b) {
                        return a->getFitness() > b->getFitness();
            });
        }


        void evolve() {
            std::vector<std::unique_ptr<Gene>> newPopulation;

            while (newPopulation.size() < individuals.size()) {
                //auto [parent1, parent2] = uniformParentSelection();
                auto [parent1, parent2] = fitnessProportionalSelection();
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
        }

};

void noAdaption() {
    Population pop(100, 10, 0.0, 10.0);
    double targetFitness = 9000;

    for (int generation = 0; generation < 10000; generation++) {
        pop.evolve();
        if (pop.getBestIndividual().getFitness() >= targetFitness) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
            break;
        }
        if (generation % 100 == 0) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
        }
    }
};


int main(void) {
    noAdaption();
    return 0;
}



