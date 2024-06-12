package main

import (
	"context"
	"github.com/saent-x/go-ml-pl/pkg/core"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// import dataset and clean the data
	dataset := "./datasets/clean-data/updated_dataset.csv"
	cleanDatasetInfo := core.ReadCSV(ctx, dataset)

	cleanDatasetInfo.SplitFeature_Target()

	// split data into train and test dataset

}
