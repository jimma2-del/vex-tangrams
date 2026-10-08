const socket = new WebSocket('/dashboard/ws');

socket.addEventListener("open", e => {
  socket.send("Hello Server!");
});

socket.addEventListener("message", e => {
  console.info("Message from server: " + e);
});


// Handle errors
socket.addEventListener("error", (event) => {
  console.error("WebSocket error:", event);
});

// Handle disconnection
socket.addEventListener("close", (event) => {
  if (event.wasClean) {
    console.log(`Closed cleanly, code=${event.code}, reason=${event.reason}`);
  } else {
    console.log("Connection died");
  }
});