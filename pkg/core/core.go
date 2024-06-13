package core

import (
	"context"
	"fmt"
	dataframe "github.com/rocketlaunchr/dataframe-go"
	"github.com/rocketlaunchr/dataframe-go/exports"
	"github.com/rocketlaunchr/dataframe-go/imports"
	"github.com/saent-x/go-ml-pl/pkg/entities"
	"github.com/sjwhitworth/golearn/base"
	"os"
	"path/filepath"
	"time"
)

type CleanedDatasetInfo struct {
	Dataframe *dataframe.DataFrame
	FileInfo  entities.FileInfo
}

func ReadCSV(ctx context.Context, path string) (CleanedDatasetInfo, error) {
	// read file
	file, err := os.Open(path)
	if err != nil {
		return CleanedDatasetInfo{}, err
	}

	defer file.Close()

	df, fileinfo, err := cleanData(ctx, file)
	if err != nil {
		return CleanedDatasetInfo{}, nil
	}
	return CleanedDatasetInfo{
		Dataframe: df,
		FileInfo:  fileinfo,
	}, nil
}

func cleanData(ctx context.Context, file *os.File) (*dataframe.DataFrame, entities.FileInfo, error) {
	// clean dataset
	df, err := imports.LoadFromCSV(ctx, file, imports.CSVLoadOptions{
		TrimLeadingSpace: false,
		LargeDataSet:     true,
		DictateDataType: map[string]interface{}{
			"timestamp":      "",
			"arbitration_id": "",
			"data_field":     "",
			"attack_0":       float64(0),
			"attack_1":       float64(0),
			"attack_2":       float64(0),
			"attack_3":       float64(0),
			"attack_4":       float64(0),
			"attack_5":       float64(0),
		},
		// TODO: specify nan values | NilValue: "NA,NaN,<nil>",
	})

	// save cleaned dataset in new file
	filename := "cleaned_dataset_" + time.Now().Format("2006-01-02") + ".csv"
	filepath := filepath.Join("datasets/session-data", filename)

	cleanedFile, err := os.Create(filepath)
	if err != nil {
		return &dataframe.DataFrame{}, entities.FileInfo{}, err
	}

	// save written data to csv
	err = exports.ExportToCSV(ctx, cleanedFile, df)
	if err != nil {
		return &dataframe.DataFrame{}, entities.FileInfo{}, err
	}

	return df, entities.FileInfo{Filename: filename, Filepath: filepath}, nil
}

func (cdi *CleanedDatasetInfo) SplitFeature_Target() {
	// deprecated: since GoLearn doesn't work with gota I'll just use their defined method for reading/parsing csv
	// switched gota to dataframe-go because dataframe-go is integrated to golearn

	cleanDataGrid := base.ConvertDataFrameToInstances(cdi.Dataframe, 0)

	// create a train dataset [Xfeatures] & [Ytargets] by splitting into features and labels
	//xfeatures, _ := base.InstancesTrainTestSplit(cleanData, -1)

	fmt.Println(cleanDataGrid)
}
