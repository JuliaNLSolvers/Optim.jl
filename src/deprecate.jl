function get_optimizer(method::Symbol)
    T = method_lookup[method]
    warn("Specifying the method using symbols is deprecated. Use \"method = $(T)()\" instead")
    T()
end
