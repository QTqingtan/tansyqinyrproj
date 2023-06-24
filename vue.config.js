
module.exports = {
    devServer: {
        hot: true,
        inline: false,

        proxy: {
            '/api/*': {
                target: 'http://127.0.0.1:5000/upload',
                // ↑这个就是你的接口地址↑
                changeOrigin: true,
                pathRewrite: {
                    '^/api': ''
                }
            }
        }

      }
}
