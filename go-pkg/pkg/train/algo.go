package train

import (
	"github.com/saent-x/go-ml-pl/pkg/core"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/ensemble"
	"github.com/sjwhitworth/golearn/evaluation"
	"golang.org/x/exp/rand"
)

func ImplRandomForests(cdi core.CleanedDatasetInfo) error {
	// read the pre-processed dataset into golearn
	// this part of the implementation is temp, TODO: to be replced with SplitFeatureTarget function
	attackDS, err := base.ParseCSVToInstances(cdi.FileInfo.Filepath, true)
	if err != nil {
		return err
	}

	// set value to seed random proceses
	rand.Seed(44111342) // TODO: revise seed value
	// no of features per tree for rf set to sqroot of attackDS features
	rf := ensemble.NewRandomForest(10, 3) // TODO: revise forestSize and features

	// Use cross-fold validation to successively train and evaluate the model
	// on 5 folds of the data set.
	cv, err := evaluation.GenerateCrossFoldValidationConfusionMatrices(attackDS /* TODO: change [attackDS] to [xfeatures]*/, rf, 10)
	if err != nil {
		return err
	}
}

func ImplXGBoost() {
}

func ImplKMeansClustering() {
}
