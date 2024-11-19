const obj = {
  array1: [1, 2, 3],
  array2: ["a", "b", "c"],
  array3: [true, false, true],
};

let isOpen = true;
let shop = (work, time, callback) => {
  setTimeout(() => {
    if (isOpen) {
      work(callback);
    } else {
      console.log("sorry for inconvience");
    }
  }, time);
};

function work(callback){
  setTimeout(()=>{
      console.log("work");
      callback();
  },2000)
}

let time = 3000;
shop(work, time, () => {
  shop(() => {
    console.log(2);
    shop(() => console.log(3), 1000, () => {});
  }, 1000, () => {});
});