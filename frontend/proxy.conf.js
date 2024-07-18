const proxy = {
    '/api': {
        context: '/api',
        target: 'http://localhost:8086',
        pathRewrite: {'^/api' : ''}
    }
};

module.exports = proxy;
