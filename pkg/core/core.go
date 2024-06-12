package core

import (
	"context"
	"github.com/saent-x/go-ml-pl/pkg/entities"
	"io"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/go-gota/gota/series"

	"github.com/go-gota/gota/dataframe"
)

type CleanedDatasetInfo struct {
	Dataframe dataframe.DataFrame
	FileInfo  entities.FileInfo
}

func ReadCSV(ctx context.Context, path string) CleanedDatasetInfo {
	// read file
	file, err := os.Open(path)
	if err != nil {
		log.Fatalln(err)
	}

	defer file.Close()

	df, fileinfo, err := cleanData(file)
	if err != nil {
		return CleanedDatasetInfo{}
	}
	return CleanedDatasetInfo{
		Dataframe: df,
		FileInfo:  fileinfo,
	}
}

func cleanData(file io.Reader) (dataframe.DataFrame, entities.FileInfo, error) {
	// clean dataset
	nanValues := []string{"NA", "NaN", "<nil>"}

	df := dataframe.ReadCSV(
		file,
		dataframe.DefaultType(series.Float),
		dataframe.DetectTypes(true),
		dataframe.HasHeader(true),
		dataframe.NaNValues(nanValues),
		dataframe.Names(
			"timestamp",
			"arbitration_id",
			"data_field",
			"attack_1",
			"attack_1",
			"attack_2",
			"attack_3",
			"attack_4",
			"attack_5",
		),
	)

	// save cleaned dataset in new file
	filename := "cleaned_dataset_" + time.Now().String()
	filepath := filepath.Join("datasets/session-data", filename)

	cleanedFile, err := os.Create(filepath)
	if err != nil {
		return dataframe.DataFrame{}, entities.FileInfo{}, err
	}

	// save written data to csv
	df.WriteCSV(cleanedFile)

	return df, entities.FileInfo{Filename: filename, Filepath: filepath}, nil
}

func (cdi *CleanedDatasetInfo) SplitFeature_Target() {
	// create a train dataset [Xfeatures] & [Ytargets] by splitting into features and labels
	//X, _
}
