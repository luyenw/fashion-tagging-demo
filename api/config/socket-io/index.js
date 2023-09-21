class Socket{
    instance = null
    constructor(){
    }
    static getInstance(){
        if(!this.instance){
            this.instance = new Socket()
        }
        return this.instance
    }
    createIO(server){
        this.io = require('socket.io')(server)
        return this.io
    }
}
module.exports = Socket