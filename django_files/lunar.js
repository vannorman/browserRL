/*
* epsioln schedule
*/
/* from
https://github.com/puresick/Lunar-Lander/
*/




var state = {
    input : {
        up : false,
        down : false,
        left : false,
        right : false
 
    }, momentum : {
        x : 0,
        y : 0,
        decay : .002,
    }, fuel : 300,
    position : {
      x : 0,
      y : 0,
    }, target : {
      x : 0,
      y : 0,
    }, gravity : 0.38,

}




$(document).keydown((key) => {
//    console.log("key rep:"+key.repeat);
switch (key.keyCode) {
    case 87: //UP (w)
      if (!state.input.down) state.input.up = true;
      break;
    case 83: //DOWN (s)
      if (!state.input.up) state.input.down = true;
      break;
    case 65: //LEFT (a)
      if (!state.input.right) state.input.left= true;
      break;
    case 68: //RIGHT (d)
      if (!state.input.left)  state.input.right = true;
      break;
  }
});


$(document).keyup((key) => {
   switch(key.keyCode){
    case 87: //UP (w)
        state.input.up = false;
    break;
    case 83: //DOWN (s)
        state.input.down = false;
      break;
    case 65: //LEFT (a)
        state.input.left = false;
      break;
    case 68: //RIGHT (d)
        state.input.right = false;
      break;
  }
 
});


// Init web socket
socket = new WebSocket(
    'ws://' + window.location.host +
    '/ws/msg/');


// receive
socket.onmessage = function(e) {
  console.log(e)
    var data = JSON.parse(e.data);
    var message = data['message'];
    
};

 socket.onclose = function(e) {
    console.error('Chat socket closed unexpectedly');
};  


  socket.onopen = () => {
  console.log('socket succesfully opened!!');
  /*
  socket.send(JSON.stringify({
      'message': message
  }));
  */
}



function GameLoop(dt){
    setTimeout(function(){GameLoop(dt);},dt);
    var power = .003;
    var consumption = .05;
    var yDir = 0;
    var xDir = 0;
  
    if (state.input.up){
        yDir = -1;
        state.momentum.y -= power;
        state.fuel = state.fuel - consumption;
    } else if (state.input.down){
        yDir = 1;
        state.momentum.y += power;
        state.fuel = state.fuel - consumption;
        
    } 
    if (state.input.right){
        xDir = 1;
        state.momentum.x += power;
        state.fuel = state.fuel - consumption;
    
    } else if (state.input.left) {
        xDir = -1;
        state.momentum.x -= power;
        state.fuel = state.fuel - consumption;
    }
    state.position.y = state.position.y + state.momentum.y;
    state.position.x = state.position.x + state.momentum.x;
    state.momentum.x = lerp(state.momentum.x, 0, state.momentum.decay);
    state.momentum.y = lerp(state.momentum.y, 0, state.momentum.decay);

    //console.log(state.input.up+","+state.input.right);
    // console.log("rew x:"+rewX);
    // var rewY = state.position.y - state.target.y < 0 ? state.momentum.y + state.gravity : -state.momentum.y - state.gravity;
    // console.log("rew y:"+rewY);
    var rew = state.position.x - state.target.x < 0 ? state.momentum.x : -state.momentum.x;

    AppendCurrentData(rew, false);
    
}

var batchSize = 500;
var batch = []
function AppendCurrentData(reward, gameWasReset){
  // for "observed data this frame"  we will convert location, fuel, momentum, into a simple array of floats.
  var obs_arr = [ state.position.x, state.position.y, state.momentum.x, state.momentum.y, state.fuel, state.target.x, state.target.y ];

  // for "input this frame" we pass a binary vector 8 with either all 0 or all 0 with one 1. e.g, [0,0,0,0,0,0,0,0] or [0,1,0,0,0,0,0,0], 
  // here [a, b, c, d, e, f, g, h] corresponding to [ up, upright, right .. etc ] or clockwise starting at 12
  var act_arr = [0,0,0,0,0,0,0,0];
  if (state.input.up && !state.input.down && !state.input.right && !state.input.left) act_arr =       [1,0,0,0,0,0,0,0];
  else if (state.input.up && !state.input.down && state.input.right && !state.input.left) act_arr =   [0,1,0,0,0,0,0,0];
  else if (!state.input.up && !state.input.down && state.input.right && !state.input.left) act_arr =  [0,0,1,0,0,0,0,0];
  else if (!state.input.up && state.input.down && state.input.right && !state.input.left) act_arr =   [0,0,0,1,0,0,0,0];
  else if (!state.input.up && state.input.down && !state.input.right && !state.input.left) act_arr =  [0,0,0,0,1,0,0,0];
  else if (!state.input.up && state.input.down && !state.input.right && state.input.left) act_arr =   [0,0,0,0,0,1,0,0];
  else if (!state.input.up && !state.input.down && !state.input.right && state.input.left) act_arr =  [0,0,0,0,0,0,1,0];
  else if (state.input.up && !state.input.down && !state.input.right && state.input.left) act_arr =   [0,0,0,0,0,0,0,1];
  
  obj = {
    'obs' : obs_arr,
    'act' : act_arr,
    'rew' : reward,
    'done' : gameWasReset
  }
  batch.push(obj);
  if (batch.length > batchSize){
    SendDataToServer();
    batch = [];
  }
}

function SendDataToServer(){
      socket.send(JSON.stringify({
          'message': batch
      }));
  // fake_data = [{'obs': np.random.rand(fake_data_obs_dim),
  //             'act': np.random.randint(2, size=fake_data_n_acts),
  //             'rew': np.random.rand(1),
  //             'done': np.random.randint(2)
  //           } for i in range(fake_data_batch_size)]
}


var lerp =  function (value1, value2, amount) {
    amount = amount < 0 ? 0 : amount;
    amount = amount > 1 ? 1 : amount;
    return value1 + (value2 - value1) * amount;
}

drawGameId = setInterval(function(){},100);

function InitTarget(){
   state.target.x =  $('#platform').offset().left;
    state.target.y =  $('#platform').offset().top; // - $('#platform').height();
}

function GameOver(status, reason){
   $('#gameover').text(reason);
    $('#gameover').stop().fadeIn(0).fadeOut(5000);
    if (status == "win"){
       AppendCurrentData(10,true);
        $('#gameover').css({'display': 'block', 'background-color': '#0f0'})
    } else {
       AppendCurrentData(-10,true);

        $('#gameover').css({'display': 'block', 'background-color': '#f00'})
    }

    ResetPosition();
}






function ResetPosition(){
   $('#lander').css({
      'top': 0,
      'left':0});
   state.position.x = 10;
   state.position.y = 10;  
   state.momentum.x = 0;
   state.momentum.y = 0;

}


$(document).ready(() => {
  InitTarget();
    GameLoop();
  function drawGame() {
    //distance calculation between player and platform

    //checking for game over cases
    if (
      state.position.y >= $(window).height()
        - parseFloat($('#lander').css('height'))
        - parseFloat($('#ground').css('height')) ||
      state.position.x >= $(window).width() - 20 ||
      state.position.x < 0 ||
      state.fuel <= 0
    ) {
      GameOver("lose","You died! (off screen)");
    };

    //checking for win case
    if (
      state.position.x - state.target.x >= 0 &&
      state.position.x - state.target.x <= parseFloat($('#platform').css('width')) &&
      Math.abs(state.position.y + $('#lander').height() - state.target.y) < 2) {
      // we landed -- but were we going to fast?
        if (Math.abs(state.momentum.x)+Math.abs (state.momentum.y) > 0.1){
            GameOver("lose","you crashed!d");
        } else {
            GAmeOver("win","You won!");

        }

    } 
    // console.log("x,y:"+state.position.x.toFixed(2)+", "+state.position.y.toFixed(2)+" -- target;"+state.target.x.toFixed(2)+", "+state.target.y.toFixed(2));
    // calculation of player movement incl. gravity and amount of state.fuel 
    $('#lander').css({
      'top': () => {
        state.position.y = state.position.y + state.gravity;
        return state.position.y;
      },
      'left': () => {
        return state.position.x;
      }
    });
    // $('#fuel').text('FUEL: ' + state.fuel.toFixed(2)); //+", momentum:"+state.momentum.x.toFixed(2)+", "+state.momentum.y.toFixed(2));
    $('#fuel').text('FUEL: ' + state.fuel.toFixed(2)+", momentum:"+state.momentum.x.toFixed(2)+", "+state.momentum.y.toFixed(2));
  };

  //drawing game at 30fps
  drawGameId = setInterval(drawGame, 0.03);
});
