const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');
const path = require('path');

const TRAIN_IMAGES_DIR = './data/train';
const TEST_IMAGES_DIR = './data/test';

function loadImages(dataDir) {
  const images = [];
  const labels = [];
  
  var files = fs.readdirSync(dataDir);
  for (let i = 0; i < files.length; i++) { 
    var filePath = path.join(dataDir, files[i]);
    
    var buffer = fs.readFileSync(filePath);
    var imageTensor = tf.node.decodeImage(buffer)
      .resizeNearestNeighbor([96,96]) // change the image size here
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims();
    images.push(imageTensor);

    var hasTuberculosis = files[i].toLocaleLowerCase().endsWith("_1.png");
    //var labelTensor = tf.tensor1d(hasTuberculosis ? tf.scalar(1) : tf.scalar(0), "int32");
    labels.push(hasTuberculosis ? 1 : 0);
  }

  return [images, labels];
}

/** Helper class to handle loading training and test data. */
class TuberculosisDataset {
  constructor() {
    this.trainData = [];
    this.testData = [];
  }

  /** Loads training and test data. */
  loadData() {
    console.log('Loading images...');
    this.trainData = loadImages(TRAIN_IMAGES_DIR);
    this.testData = loadImages(TEST_IMAGES_DIR);
    console.log('Images loaded successfully.')
  }

  getTrainData() {
    return {
      images: tf.concat(this.trainData[0]),
      labels: tf.oneHot(tf.tensor1d(this.trainData[1], 'int32'), 2).toFloat()
    }
  }

  getTestData() {
    return {
      images: tf.concat(this.testData[0]),
      labels: tf.oneHot(tf.tensor1d(this.testData[1], 'int32'), 2).toFloat()
    }
  }
}

module.exports = new TuberculosisDataset();
