package main

import (
	"context"
	"fmt"

	"log"

	// "github.com/saent-x/go-ml-pl/pkg/core"
	// "github.com/saent-x/go-ml-pl/pkg/train"
	"go_pkg/pkg/core"
	"go_pkg/pkg/train"
	"mlpack.org/v1/mlpack"
)

func main() {
	_, cancel := context.WithCancel(context.Background())c
	defer cancel()

	// import dataset and clean the data
	dataset := "./datasets/clean-data/updated_dataset.csv"

	// split data into feature and target subsets
	fmt.Println("==> ml-algo [Initializing]")

	training_set, err := core.SplitFeature_Target(dataset)
	if err != nil {
		log.Printf("==> ml-algo [%s]\n", err.Error())
	}

	// Implement Random Forests and Generate confusion matrix
	cv, err := train.ImplRandomForests(training_set)
	if err != nil {
		log.Printf("==> ml-algo [%s]\n", err.Error())
	}

	fmt.Printf("==> ml-algo [%+v]\n", cv)
}
