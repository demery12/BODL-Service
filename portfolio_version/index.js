$(".myButton").click(function(){
    fetch("big_drink_list.json")
        .then(function (response) {
            return response.json()
        })
        .then(function (data) {
            writeRecipes(data)
        })
        .catch(function (err) {
            console.log(err);
        });
});

function writeRecipes(data) {
    let index = randomIntFromInterval(0, data.length - 1);
    let ingredients = data[index].ingredients.map(ingredient => "<br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-" + ingredient).join();
    $("#result").html("<br><b>" + data[index].drink_name + "</b>"+ ingredients);
}

function randomIntFromInterval(min, max) { // min and max included
  return Math.floor(Math.random() * (max - min + 1) + min);
}