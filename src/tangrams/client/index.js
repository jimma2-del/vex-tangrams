// const socket = new WebSocket('/dashboard/ws');

// socket.addEventListener("open", e => {
//   socket.send("Hello Server!");
// });

// socket.addEventListener("message", e => {
//   console.info("Message from server: " + e);
// });


// // Handle errors
// socket.addEventListener("error", (event) => {
//   console.error("WebSocket error:", event);
// });

// // Handle disconnection
// socket.addEventListener("close", (event) => {
//   if (event.wasClean) {
//     console.log(`Closed cleanly, code=${event.code}, reason=${event.reason}`);
//   } else {
//     console.log("Connection died");
//   }
// });


let TriangleisDragging = false;
let rectangleisDragging = false;
console.log("hii");
const triangle = {
  x: 300,
  y: 300,
  color: '#c72614',
  points: [{x:0,y:-25},{x:150,y:150},{x:-150,y:150}]
};
const rectangle = {
  x: 500,
  y: 500,
  color: '#3a44c4',
  points: [{x:0,y:0},{x:100,y:0},{x:100,y:-100},{x:0,y:-100}]
};
const bigtriangle = {
  x: 200,
  y: 200,
  color: '#21c578',
  points: [{x:-70,y:-150},{x:-70,y:150},{x:70,y:0}]
};
const topbigtriangle = {  
  x: 270,
  y: 125,
  color: '#eb3a8d',
  points: [{x:-100,y:-150},{x:-100,y:150},{x:50,y:0}]
};


const canvas = document.getElementById("tangramCanvas");
const ctx = canvas.getContext("2d");


function draw(){
  ctx.clearRect(0,0,900,900);
  ctx.beginPath();
  ctx.fillStyle = triangle.color;
  ctx.moveTo(triangle.x + triangle.points[0].x, triangle.y + triangle.points[0].y);
  ctx.lineTo(triangle.x + triangle.points[1].x, triangle.y + triangle.points[1].y);
  ctx.lineTo(triangle.x + triangle.points[2].x, triangle.y + triangle.points[2].y);
  ctx.lineTo(triangle.x + triangle.points[0].x, triangle.y + triangle.points[0].y);
  ctx.fill();

  //rectangle
  ctx.beginPath();
  ctx.fillStyle = rectangle.color;
  ctx.moveTo(rectangle.x + rectangle.points[0].x, rectangle.y + rectangle.points[0].y);
  ctx.lineTo(rectangle.x + rectangle.points[1].x, rectangle.y + rectangle.points[1].y);
  ctx.lineTo(rectangle.x + rectangle.points[2].x, rectangle.y + rectangle.points[2].y);
  ctx.lineTo(rectangle.x + rectangle.points[3].x, rectangle.y + rectangle.points[3].y);
  ctx.lineTo(rectangle.x + rectangle.points[0].x, rectangle.y + rectangle.points[0].y);
  ctx.fill();

  ctx.beginPath();
  ctx.fillStyle = bigtriangle.color;
  ctx.moveTo(bigtriangle.x + bigtriangle.points[0].x, bigtriangle.y + bigtriangle.points[0].y);
  ctx.lineTo(bigtriangle.x + bigtriangle.points[1].x, bigtriangle.y + bigtriangle.points[1].y);
  ctx.lineTo(bigtriangle.x + bigtriangle.points[2].x, bigtriangle.y + bigtriangle.points[2].y);
  ctx.lineTo(bigtriangle.x + bigtriangle.points[0].x, bigtriangle.y + bigtriangle.points[0].y);
  ctx.fill();
}

draw();

function move(){

}

function TMouseDown(event) {
  let rect = canvas.getBoundingClientRect();
  const mouseX = event.clientX - rect.left;
  const mouseY = event.clientY - rect.top;
  console.log(mouseX + " " + mouseY);
  const distance_triangle = Math.hypot(mouseX - triangle.x, mouseY - triangle.y);
  const distance_rectangle =  Math.hypot(mouseX - (rectangle.x + 50), mouseY - (rectangle.y-50));
  if (distance_triangle < 100) {
    console.log("hoi");
    TriangleisDragging = true;
  }
  if (distance_rectangle < 50) {
    console.log("hoi");
    rectangleisDragging = true;
  }
}

function TMouseMove(event) {
  let rect = canvas.getBoundingClientRect();
  if (TriangleisDragging) {
    triangle.x = event.clientX - rect.left;
    triangle.y = event.clientY - rect.top;
    draw();
  }
  if (rectangleisDragging) {
    rectangle.x = event.clientX - rect.left - 50;
    rectangle.y = event.clientY - rect.top + 50;
    draw();
  }
}

function TMouseUp() {
  TriangleisDragging = false;
  rectangleisDragging = false;
}
function rotation(){
  if(TriangleisDragging){}
}

canvas.addEventListener("mousedown", TMouseDown);
canvas.addEventListener("mousemove", TMouseMove);
canvas.addEventListener("mouseup", TMouseUp);




