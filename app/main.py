import Replicate from "replicate";

const replicate = new Replicate({
  auth: process.env.RKEY,
});

let prediction = await replicate.deployments.predictions.create(
  "constantinbender51-cmyk",
  "deepseek-base",
  {
    input: {
      prompt: "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
    }
  }
);
prediction = await replicate.wait(prediction);
console.log(prediction.output);