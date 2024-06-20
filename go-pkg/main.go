package main

import (
	"context"
	"github.com/saent-x/go-ml-pl/pkg/core"
	"log"
)

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// import dataset and clean the data
	dataset := "./datasets/clean-data/updated_dataset.csv"
	cleanDatasetInfo, err := core.ReadCSV(ctx, dataset)
	if err != nil {
		log.Println(err.Error())
	}
	// fmt.Println(cleanDatasetInfo.FileInfo)

	// split data into feature and target subsets
	cleanDatasetInfo.SplitFeature_Target()
}
