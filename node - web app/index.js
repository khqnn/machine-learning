const express = require('express')
const ort = require('onnxruntime-node');

const app = express()
const port = 3000

const ingredientsVector = [
    {
        value: 4,
        name: 'Apple',
        quantity: 1
    }
]

const InferenceSession = ort.InferenceSession;

app.get('/predict', async (req, res) => {




    const ingredients = Array(380).fill(0);
    const session = await InferenceSession.create('../classification/model.onnx');



    ingredientsVector.forEach(ingredient => {
        ingredients[ingredient.value] = ingredient.quantity
    });

    const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
    const feeds = { float_input: input };

    const results = await session.run(feeds);


    res.json({ data: results })
})

app.listen(port, () => {
    console.log(`Started listening at http://localhost:${port}`);
})