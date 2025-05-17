package main

import (
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func main() {
	inputData := []float64{0, 0, 0, 1, 1, 0, 1, 1}
	inputs := mat.NewDense(4, 2, inputData)

	labels := mat.NewDense(4, 1, []float64{0, 1, 1, 1})

	config := neuralNetConfig{
		inputNeurons:  2,
		outputNeurons: 1,
		numEpochs:     20000,
		learningRate:  1,
	}

	network := newNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}
}

// neuralNet contains all of the information
// that defines a trained neural network.
type neuralNet struct {
	config neuralNetConfig
	wOut   *mat.Dense
	bOut   *mat.Dense
}

// neuralNetConfig defines our neural network
// architecture and learning parameters.
type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	numEpochs     int
	learningRate  float64
}

// newNetwork initializes a new neural network.
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// sigmoid implements the sigmoid function
// for use in activation functions.
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrime implements the derivative
// of the sigmoid function for backpropagation.
func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
	// return x * (1.0 - x)
}

// train trains a neural network using backpropagation.
func (nn *neuralNet) train(x, y *mat.Dense) error {

	// Initialize biases/weights.
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wOut := mat.NewDense(nn.config.inputNeurons, nn.config.outputNeurons, nil)
	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// Define the output of the neural network.
	output := new(mat.Dense)

	// Use backpropagation to adjust the weights and biases.
	if err := nn.backpropagate(x, y, wOut, bOut, output); err != nil {
		return err
	}

	// Define our trained neural network.
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

// backpropagate completes the backpropagation method.
func (nn *neuralNet) backpropagate(x, y, wOut, bOut, output *mat.Dense) error {

	// Loop over the number of epochs utilizing
	// backpropagation to train our model.
	for i := 0; i < nn.config.numEpochs; i++ {

		// Complete the feed forward process.
		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(x, wOut)
		addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
		outputLayerInput.Apply(addBOut, outputLayerInput)

		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		output.Apply(applySigmoid, outputLayerInput)

		// Complete the backpropagation.
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)

		// Adjust the parameters.
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(x.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		if i%10 == 0 {
			// calculate error
			allErr := networkError.RawMatrix().Data
			score := 0.0
			for i := 0; i < len(allErr); i++ {
				score += math.Pow(allErr[i], 2)
			}
			score = math.Pow(score/float64(len(allErr)), 0.5)

			fmt.Printf("Epoch: %d\n", i)
			fmt.Printf("Error Score: %v\n", score)
			fmt.Printf("Weights: %v\n", wOut.RawMatrix().Data)
			fmt.Printf("Bias: %v\n", bOut.RawMatrix().Data)
		}
	}

	return nil
}

// sumAlongAxis sums a matrix along a particular dimension,
// preserving the other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}
