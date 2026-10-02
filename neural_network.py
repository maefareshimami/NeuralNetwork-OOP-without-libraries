import layer


class NeuralNetwork:

    def __init__(self, name, nb_layers, layers_scheme):
        self.name = name
        self.nb_layers = nb_layers   #  Entry and Exit included
        self.layers_scheme = layers_scheme
        #self.entries = entries
        self.list_layers = [layer.FirstLayer(self.layers_scheme[0], self.layers_scheme[0])] + [layer.Layer(i, self.layers_scheme[i], self.layers_scheme[i - 1]) for i in range(1, self.nb_layers - 1)] + [layer.LastLayer(nb_layers - 1, self.layers_scheme[-1], self.layers_scheme[-2])]
        
    def __repr__(self):
        return f"<class NeuralNetwork | Name: {self.name} | Number of layers: {self.nb_layers} | Height Scheme: {self.layers_scheme}>"
    
    """
    def forwardStep(self):
        #Compute y with x as entry
        current_layers = self.list_layers
        current_entries = self.entries
        current_layer = current_layers[0]
        current_layer.inputs = current_entries
        current_layer.computeOutputsFirstLayer()
        current_layer.computeActivationFunctionFirstLayer()
        current_entries = current_layer.act_func
        for i in range(1, self.nb_layers - 1):
            current_layer = current_layers[i]
            current_layer.inputs = current_entries
            current_layer.computeOutputs()
            current_layer.computeActivationFunction()
            current_entries = current_layer.act_func
        current_layer = current_layers[self.nb_layers - 1]
        current_layer.inputs = current_entries
        current_layer.computeOutputsLastLayer()
        current_layer.computeActivationFunctionLastLayer()
        current_entries = current_layer.act_func
        return current_entries
    """