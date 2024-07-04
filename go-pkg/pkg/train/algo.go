package train

import (
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
	"golang.org/x/exp/rand"
)

func ImplRandomForests(training_set *base.DenseInstances) ([]evaluation.ConfusionMatrix, error) {

	// set value to seed random proceses
	rand.Seed(44111342) // TODO: revise seed value

	// no of features per tree for rf set to sqroot of attackDS features
	rf := ensemble.NewRandomForest(10, 3) // TODO: revise forestSize and features

	// train model on training dataset and generate confusion matrix
	cv, err := evaluation.GenerateCrossFoldValidationConfusionMatrices(training_set, rf, 10)
	if err != nil {
		return nil, err
	}

	return cv, nil
}

func ImplXGBoost() {
}

func ImplKMeansClustering() {
}
