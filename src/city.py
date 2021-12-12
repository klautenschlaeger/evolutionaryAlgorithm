class City:
    def __init__(self, id, neighbours, distances):
        self.id = id
        self.distances = {}
        self.__setDistances(neighbours=neighbours, distances=distances)

    def __setDistances(self, neighbours, distances):
        for index, city in enumerate(neighbours):
            self.distances[city] = distances[index]

    def distance(self, city, penalty):
        if city.id in self.distances:
            return self.distances[city.id]
        else:
            return penalty

    def __repr__(self):
        return "(" + str(self.id) + ")"