FROM golang:1.22.4
LABEL authors="saent001"

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY /datasets/clean-data ./
COPY /pkg ./
COPY *.go ./
COPY *.py ./
COPY *.ipynb ./

RUN go build -o /auto-bin

EXPOSE 5050

CMD ["/auto-bin"]