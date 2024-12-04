#include <cmath>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
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
    virtual double cosineSimilarity(const Gene& other) const = 0;
    virtual std::unique_ptr<Gene> clone() const = 0;

    virtual std::pair<std::unique_ptr<Gene>, std::unique_ptr<Gene>>
        crossover(const Gene& other) const = 0;
    double getFitness() const { return fitness; }
    void setMutationRate(double mutationRate) { this->mutationRate = mutationRate; }
    void scaleMutationRate(double scale) { mutationRate *= scale; }
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
            fitness = oneMaxFunction();
        }

        double oneMaxFunction() {
            double sum = 0;
            std::for_each(this->alleles.begin(), this->alleles.end(), [&] (double n) {
                sum += n;
            });
            return sum;
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

        double cosineSimilarity(const Gene& other) const override {
            const BitGene& otherBit = dynamic_cast<const BitGene&>(other);
            double dotProduct = 0;
            double magnitudeA = 0;
            double magnitudeB = 0;

            for (int i = 0; i < alleles.size(); i++) {
                dotProduct += this->getAllele(i) * otherBit.getAllele(i);
                magnitudeA += this->getAllele(i) * this->getAllele(i);
                magnitudeB += otherBit.getAllele(i) * otherBit.getAllele(i);
            }

            magnitudeA = std::sqrt(magnitudeA);
            magnitudeB = std::sqrt(magnitudeB);

            if (magnitudeA == 0 || magnitudeB == 0) {
                //TODO: Define a way to handle such case.
                throw std::invalid_argument("One of the vector has zero magnitude");
            }

            return dotProduct / (magnitudeA * magnitudeB);
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
            fitness = sphereFunction();
        }

        double sphereFunction() {
            double sum = 0;
            std::for_each(this->alleles.begin(), this->alleles.end(), [&] (double n) {
                sum += n * n;
            });
            return sum;
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

        double cosineSimilarity(const Gene& other) const override {
            const IntGene& otherInt = dynamic_cast<const IntGene&>(other);
            double dotProduct = 0;
            double magnitudeA = 0;
            double magnitudeB = 0;

            for (int i = 0; i < alleles.size(); i++) {
                dotProduct += this->getAllele(i) * otherInt.getAllele(i);
                magnitudeA += this->getAllele(i) * this->getAllele(i);
                magnitudeB += otherInt.getAllele(i) * otherInt.getAllele(i);
            }

            magnitudeA = std::sqrt(magnitudeA);
            magnitudeB = std::sqrt(magnitudeB);

            if (magnitudeA == 0 || magnitudeB == 0) {
                //TODO: Define a way to handle such case.
                throw std::invalid_argument("One of the vector has zero magnitude");
            }
            return dotProduct / (magnitudeA * magnitudeB);
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
            //fitness = sphereFunction();
            fitness = 2000 - RastriginFunction();
        }

        double sphereFunction() {
            double sum = 0;
            std::for_each(this->alleles.begin(), this->alleles.end(), [&] (double n) {
                sum += n * n;
            });
            return sum;
        }

        double RastriginFunction() {
            double sum = 0;
            std::for_each(alleles.begin(), alleles.end(), [&] (double x) {
                sum += x * x;
                sum -= 10 * cos(2 * M_PI * x);
            });
            sum += 10 * alleles.size();
            return sum;
        }

        double RosenbrockFunction() {
            int n = alleles.size();
            double sum = 0;
            double temp;
            for(int i = 0; i < n - 1; i++) {
                temp = 100 * std::pow(alleles[i] * alleles[i] - alleles[i + 1], 2);
                sum += temp + std::pow(alleles[i] - 1, 2);
            }
            return sum;
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
            double temp;
            for (int i = 0; i < this->alleles.size(); i++) {
                if (dis(gen) < this->mutationRate) {
                    temp = alleles[i] += disR(gen);
                    temp = std::min(temp, maxAllele);
                    temp = std::max(temp, minAllele);
                    this->alleles[i] = temp;
                }
            }
        }

        void setAllele(int idx, double val) {
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

        double cosineSimilarity(const Gene& other) const override {
            const RealGene& otherReal = dynamic_cast<const RealGene&>(other);
            double dotProduct = 0;
            double magnitudeA = 0;
            double magnitudeB = 0;

            for (int i = 0; i < alleles.size(); i++) {
                dotProduct += this->getAllele(i) * otherReal.getAllele(i);
                magnitudeA += this->getAllele(i) * this->getAllele(i);
                magnitudeB += otherReal.getAllele(i) * otherReal.getAllele(i);
            }

            magnitudeA = std::sqrt(magnitudeA);
            magnitudeB = std::sqrt(magnitudeB);

            if (magnitudeA == 0 || magnitudeB == 0) {
                //TODO: Define a way to handle such case.
                throw std::invalid_argument("One of the vector has zero magnitude");
            }
            return dotProduct / (magnitudeA * magnitudeB);
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
        int populationSize;
        int offspringSize;
        double diversity;
        double populationSimilarity;
        double averageFitness;
        double totalFitness;
        double minFitness;
        double maxFitness;
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
        Population(int size, int n, int offspringSize = -1) {
            for (int i = 0; i < size; i++) {
                auto gene = createGene(n);
                individuals.push_back(std::move(gene));
            }
            populationSize = size;
            this->offspringSize = offspringSize;
            diversity = calculateDiveristy();
            populationSimilarity = calculateSimilarity();
            calculatePopulationFitnessMetrics();
        }

        Population(int size, int n, int minAllele, int maxAllele, int offspringSize = -1) {
            for (int i = 0; i < size; i++) {
                auto gene = createIntGene(n, minAllele, maxAllele);
                individuals.push_back(std::move(gene));
            }
            populationSize = size;
            this->offspringSize = offspringSize;
            diversity = calculateDiveristy();
            populationSimilarity = calculateSimilarity();
            calculatePopulationFitnessMetrics();
        }

        Population(int size, int n, double minAllele, double maxAllele, int offspringSize = -1) {
            for (int i = 0; i < size; i++) {
                auto gene = createRealGene(n, minAllele, maxAllele);
                individuals.push_back(std::move(gene));
            }
            populationSize = size;
            this->offspringSize = offspringSize;
            diversity = calculateDiveristy();
            populationSimilarity = calculateSimilarity();
            calculatePopulationFitnessMetrics();
        }

        int getPopulationSize() const { return populationSize; }
        int getOffspringSize() const { return offspringSize; }
        double getAverageFitness() const { return averageFitness; }

        std::pair<const Gene*, const Gene*> uniformParentSelection() const {
            std::uniform_int_distribution<>dis(0, individuals.size() - 1);
            return {individuals[dis(gen)].get(), individuals[dis(gen)].get()};
        }

        std::pair<const Gene*, const Gene*> fitnessProportionalSelection() const {
            //TODO: Dunno why this won't work for subtract the fitness
            //      This is working well for int type gene...
            //std::uniform_real_distribution<> dis(0.0, totalFitness - minFitness * populationSize);
            std::uniform_real_distribution<> dis(0.0, totalFitness);
            double rnd1 = dis(gen);
            double rnd2 = dis(gen);

            const Gene* parent1 = nullptr;
            const Gene* parent2 = nullptr;
            double accumulatedFitness = 0.0;
            for(const auto& individual : individuals) {
                accumulatedFitness += individual->getFitness();
                //TODO: Fix this later....
                //      It return nullptr for parent...
                //accumulatedFitness -= minFitness;
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

        double getDivsersity() const { return diversity; }

        double calculateDiveristy() {
            // pairwise distance average
            double totalDistance = 0.0;
            int pairs = individuals.size() * (individuals.size() - 1) / 2;
            for (int i = 0; i < individuals.size(); i++) {
                for (int j = i + 1; j < individuals.size(); j++) {
                    totalDistance += individuals[i]->calculateDistance(*individuals[j]);
                }
            }
            this->diversity = totalDistance / pairs;
            return totalDistance / pairs;
        }

        double calculateSimilarity() {
            double totalSimilarity = 0.0;
            int pairs = individuals.size() * (individuals.size() - 1) / 2;
            for (int i = 0; i < individuals.size(); i++) {
                for (int j = i + 1; j < individuals.size(); j++) {
                    totalSimilarity += individuals[i]->cosineSimilarity(*individuals[j]);
                }
            }
            this->populationSimilarity = totalSimilarity / pairs;
            return totalSimilarity / pairs;
        }

        void calculatePopulationFitnessMetrics() {
            double sum = 0.0;
            double minFitness = std::numeric_limits<double>::max();
            double maxFitness = std::numeric_limits<double>::min();

            for (const auto& individual : individuals) {
                sum += individual->getFitness();
                minFitness = std::min(minFitness, individual->getFitness());
                maxFitness = std::max(maxFitness, individual->getFitness());
            }

            this->maxFitness = maxFitness;
            this->minFitness = minFitness;
            this->totalFitness = sum;
            this->averageFitness = sum / individuals.size();
        }

        void sortPopulationWithFitness() {
            std::sort(individuals.begin(), individuals.end(),
                    [](const std::unique_ptr<Gene>& a, const std::unique_ptr<Gene>& b) {
                        return a->getFitness() > b->getFitness();
            });
        }

        void setPopulationMutationRate(double mutationRate) {
            for(auto& individual : individuals) {
                individual->setMutationRate(mutationRate);
            }
        }

        void scalePopulationMutationRate(double scale) {
            for(auto& individual : individuals) {
                individual->scaleMutationRate(scale);
            }
        }

        void scalePopulationMutationRate(double scale, double lb, double ub) {
            for(auto& individual : individuals) {
                double mutationRate = individual->getMutationRate() * scale;
                mutationRate = std::max(mutationRate, lb);
                mutationRate = std::min(mutationRate, ub);
                individual->setMutationRate(mutationRate);
            }
        }

        void selectNextPopulationWithoutReplacement() {
            std::vector<std::unique_ptr<Gene>> newPopulation;
            // Random Selection
            shuffle(individuals.begin(), individuals.end(), gen);
            for(int i = 0; i < populationSize; i++){
                newPopulation.push_back(std::move(individuals[i]));
            }
            individuals = std::move(newPopulation);
        }

        void selectNextPopulationWithElitismWithoutReplacement() {
            std::vector<std::unique_ptr<Gene>> newPopulation;
            sortPopulationWithFitness();
            for(int i = 0; i < populationSize; i++){
                newPopulation.push_back(std::move(individuals[i]));
            }
            individuals = std::move(newPopulation);
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
            individuals.insert(
                individuals.end(),
                std::make_move_iterator(newPopulation.begin()),
                std::make_move_iterator(newPopulation.end())
            );
            //selectNextPopulationWithoutReplacement();
            selectNextPopulationWithElitismWithoutReplacement();
            calculatePopulationFitnessMetrics();
        }

        int evolveWithSuccessMutation() {
            std::vector<std::unique_ptr<Gene>> newPopulation;
            int successCount = 0;
            double fitness1;
            double fitness2;
            while (newPopulation.size() < individuals.size()) {
                //auto [parent1, parent2] = uniformParentSelection();
                auto [parent1, parent2] = fitnessProportionalSelection();
                auto [offspring1, offspring2] = parent1->crossover(*parent2);
                fitness1 = offspring1->getFitness();
                fitness2 = offspring2->getFitness();
                offspring1->mutate();
                offspring2->mutate();
                offspring1->calculateFitness();
                offspring2->calculateFitness();
                if (offspring1->getFitness() > fitness1) successCount++;
                if (offspring2->getFitness() > fitness2) successCount++;
                newPopulation.push_back(std::move(offspring1));
                if(newPopulation.size() < individuals.size()) {
                    newPopulation.push_back(std::move(offspring2));
                }
            }
            individuals.insert(
                individuals.end(),
                std::make_move_iterator(newPopulation.begin()),
                std::make_move_iterator(newPopulation.end())
            );
            //selectNextPopulationWithoutReplacement();
            selectNextPopulationWithElitismWithoutReplacement();
            calculatePopulationFitnessMetrics();
            return successCount;
        }

        void evolveWithMatingDistance() {
            std::vector<std::unique_ptr<Gene>> newPopulation;
            int count = 0;
            while (newPopulation.size() < individuals.size()) {
                //auto [parent1, parent2] = uniformParentSelection();
                auto [parent1, parent2] = fitnessProportionalSelection();
                if(parent1->cosineSimilarity(*parent2) < 0.3 && count < 20) {
                    count++;
                    continue;
                }
                count = 0;
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
            individuals.insert(
                individuals.end(),
                std::make_move_iterator(newPopulation.begin()),
                std::make_move_iterator(newPopulation.end())
            );
            //selectNextPopulationWithoutReplacement();
            selectNextPopulationWithElitismWithoutReplacement();
            calculatePopulationFitnessMetrics();
        }
};

void noAdaption(Population& pop, double targetFitness, int maxGenerations) {
    for (int generation = 0; generation < maxGenerations; generation++) {
        pop.evolve();
        if (pop.getBestIndividual().getFitness() >= targetFitness) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
            break;
        }
        if (generation % 10000 == 0) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
        }
    }
};

void oneFifthSuccessRule(Population& pop, double targetFitness, int maxGenerations) {
    double c = 0.85; // 0.817 <= c <= 1 is suggested
    int successCount = 0;
    int nPeriod = 20;

    for (int generation = 0; generation < maxGenerations; generation++) {
        successCount += pop.evolveWithSuccessMutation();
        if (pop.getBestIndividual().getFitness() >= targetFitness) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
            break;
        }
        if (generation % 10000 == 0) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
        }
        if (generation && generation % nPeriod == 0) {
            if (successCount * 5 > nPeriod * pop.getPopulationSize()) {
                pop.scalePopulationMutationRate(1.0 / c, 0.01, 0.9);
            } else if (successCount * 5 < nPeriod * pop.getPopulationSize()) {
                pop.scalePopulationMutationRate(c, 0.01, 0.9);
            }
            successCount = 0;
        }
    }
};


void diversityControl(Population& pop, double targetFitness, int maxGenerations) {
    double previousDiversity = pop.calculateDiveristy();
    double currentDiversity = 0;
    int nPeriod = 20;

    for (int generation = 0; generation < maxGenerations; generation++) {
        pop.evolve();
        currentDiversity += pop.calculateDiveristy();
        if (pop.getBestIndividual().getFitness() >= targetFitness) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
            break;
        }
        if (generation % 10000 == 0) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
        }
        if (generation && generation % nPeriod == 0) {
            currentDiversity = currentDiversity / nPeriod;
            if (currentDiversity > previousDiversity) {
                pop.scalePopulationMutationRate(0.99, 0.01, 0.9);
            } else if (currentDiversity < previousDiversity) {
                pop.scalePopulationMutationRate(1.01, 0.01, 0.9);
            }
            previousDiversity = currentDiversity;
            currentDiversity = 0;
        }
    }
};

void similarityCompare(Population& pop, double targetFitness, int maxGenerations) {
    double previousSimilarity = pop.calculateSimilarity();
    double currentSimilarity = 0;
    int nPeriod = 20;

    for (int generation = 0; generation < maxGenerations; generation++) {
        pop.evolve();
        currentSimilarity += pop.calculateDiveristy();
        if (pop.getBestIndividual().getFitness() >= targetFitness) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
            break;
        }
        if (generation % 10000 == 0) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
        }
        if (generation && generation % nPeriod == 0) {
            currentSimilarity = currentSimilarity / nPeriod;
            if (currentSimilarity > previousSimilarity) {
                pop.scalePopulationMutationRate(0.99, 0.01, 0.9);
            } else if (currentSimilarity < previousSimilarity) {
                pop.scalePopulationMutationRate(1.01, 0.01, 0.9);
            }
            previousSimilarity = currentSimilarity;
            currentSimilarity = 0;
        }
    }
};

void similarityControl(Population& pop, double targetFitness, int maxGenerations) {
    double currentSimilarity = 0;
    int nPeriod = 20;

    for (int generation = 0; generation < maxGenerations; generation++) {
        pop.evolve();
        currentSimilarity += pop.calculateDiveristy();
        if (pop.getBestIndividual().getFitness() >= targetFitness) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
            break;
        }
        if (generation % 10000 == 0) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
        }
        if (generation && generation % nPeriod == 0) {
            currentSimilarity = currentSimilarity / nPeriod;
            if (currentSimilarity > 0.8) {
                pop.scalePopulationMutationRate(0.9, 0.01, 0.9);
            } else if (currentSimilarity < 0.4) {
                pop.scalePopulationMutationRate(1.0 / 0.9, 0.01, 0.9);
            }
            currentSimilarity = 0;
        }
    }
};

void cosineAnneling(Population& pop, double targetFitness, int maxGenerations) {
    double etaMin = 0.0;
    double etaMax = 0.5;
    double etaCur = etaMax;
    for (int generation = 0; generation < maxGenerations; generation++) {
        pop.evolve();
        if (pop.getBestIndividual().getFitness() >= targetFitness) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
            break;
        }
        if (generation % 10000 == 0) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
        }
        etaCur = etaMin + (etaMax - etaMin) * ( 1 + std::cos(M_PI * generation / maxGenerations)) / 2;
        pop.setPopulationMutationRate(etaCur);
    }
};

void averageFitness(Population& pop, double targetFitness, int maxGenerations) {
    double c = 0.85; // 0.817 <= c <= 1 is suggested
    double previousAverageFitness = pop.getAverageFitness();
    double currentAverageFitness = 0;
    int nPeriod = 20;

    for (int generation = 0; generation < maxGenerations; generation++) {
        pop.evolve();
        currentAverageFitness += pop.getAverageFitness();
        if (pop.getBestIndividual().getFitness() >= targetFitness) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
            break;
        }
        if (generation % 10000 == 0) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
        }
        if (generation && generation % nPeriod == 0) {
            currentAverageFitness /= nPeriod;
            if (currentAverageFitness > previousAverageFitness) {
                pop.scalePopulationMutationRate(1 / c, 0.01, 0.9);
            } else if (currentAverageFitness < previousAverageFitness) {
                pop.scalePopulationMutationRate(c, 0.01, 0.9);
            }
            previousAverageFitness = currentAverageFitness;
            currentAverageFitness = 0;
        }
    }
};

void matingDistance(Population& pop, double targetFitness, int maxGenerations) {
    for (int generation = 0; generation < maxGenerations; generation++) {
        pop.evolveWithMatingDistance();
        if (pop.getBestIndividual().getFitness() >= targetFitness) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
            break;
        }
        if (generation % 10000 == 0) {
            std::cout << "Generation " << generation << ":\n";
            std::cout << "Best Fitness: " << pop.getBestIndividual().getFitness() << "\n";
            std::cout << "Average Fitness: " << pop.getAverageFitness() << "\n";
            std::cout << "Best Individual: " << pop.getBestIndividual() << "\n\n";
        }
    }
};

int main(void) {
    int populationSize = 100;
    int allelesLength = 20;
    double minAllele = 0.0;
    double maxAllele = 10.0;
    double targetFitness = 1980;
    int maxGenerations = 1e6;
    double initMutationRate = 0.05;
    int N = 10;
    Population pop(1, 1);

    std::cout << "Standard EC\n";
    for(int i = 0; i < N; i++) {
        std::cout << "Run #" << i + 1 << "\n";
        pop = Population(populationSize, allelesLength, minAllele, maxAllele);
        pop.setPopulationMutationRate(initMutationRate);
        noAdaption(pop, targetFitness, maxGenerations);
    }
    std::cout << "\n\n";

    std::cout << "1/5 Success Rule\n";
    for(int i = 0; i < N; i++) {
        std::cout << "Run #" << i + 1 << "\n";
        pop = Population(populationSize, allelesLength, minAllele, maxAllele);
        pop.setPopulationMutationRate(initMutationRate);
        oneFifthSuccessRule(pop, targetFitness, maxGenerations);
    }
    std::cout << "\n\n";

    std::cout << "Average Pairwise Distance\n";
    for(int i = 0; i < N; i++) {
        std::cout << "Run #" << i + 1 << "\n";
        pop = Population(populationSize, allelesLength, minAllele, maxAllele);
        pop.setPopulationMutationRate(initMutationRate);
        diversityControl(pop, targetFitness, maxGenerations);
    }
    std::cout << "\n\n";

    std::cout << "Cosine Similarity with Compare\n";
    for(int i = 0; i < N; i++) {
        std::cout << "Run #" << i + 1 << "\n";
        pop = Population(populationSize, allelesLength, minAllele, maxAllele);
        pop.setPopulationMutationRate(initMutationRate);
        similarityCompare(pop, targetFitness, maxGenerations);
    }
    std::cout << "\n\n";

    std::cout << "Cosine Similarity with Threshold\n";
    for(int i = 0; i < N; i++) {
        std::cout << "Run #" << i + 1 << "\n";
        pop = Population(populationSize, allelesLength, minAllele, maxAllele);
        pop.setPopulationMutationRate(initMutationRate);
        similarityControl(pop, targetFitness, maxGenerations);
    }
    std::cout << "\n\n";

    std::cout << "Cosine Annealing\n";
    for(int i = 0; i < N; i++) {
        std::cout << "Run #" << i + 1 << "\n";
        pop = Population(populationSize, allelesLength, minAllele, maxAllele);
        pop.setPopulationMutationRate(initMutationRate);
        similarityControl(pop, targetFitness, maxGenerations);
    }
    std::cout << "\n\n";

    std::cout << "Average Fitness\n";
    for(int i = 0; i < N; i++) {
        std::cout << "Run #" << i + 1 << "\n";
        pop = Population(populationSize, allelesLength, minAllele, maxAllele);
        pop.setPopulationMutationRate(initMutationRate);
        averageFitness(pop, targetFitness, maxGenerations);
    }
    std::cout << "\n\n";

    std::cout << "Mating Distance\n";
    for(int i = 0; i < N; i++) {
        std::cout << "Run #" << i + 1 << "\n";
        pop = Population(populationSize, allelesLength, minAllele, maxAllele);
        pop.setPopulationMutationRate(initMutationRate);
        matingDistance(pop, targetFitness, maxGenerations);
    }
    std::cout << "\n\n";
    return 0;
}

